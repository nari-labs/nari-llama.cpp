#include "llama.h"
#include "ggml-backend.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;
using clock_type = std::chrono::steady_clock;

static void check(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

static double seconds(clock_type::time_point start) {
    return std::chrono::duration<double>(clock_type::now() - start).count();
}

// Manual, bounded full-model check. Not registered with CTest and never downloads weights.
int main(int argc, char ** argv) {
    try {
        std::string model_path, output, force_path;
        std::string prompt = "The capital of France is";
        int ngl = 0, ubatch = 8, predict = 8, threads = 6;
        for (int i = 1; i < argc; i += 2) {
            check(i + 1 < argc, "options require values");
            const std::string key = argv[i], value = argv[i + 1];
            if (key == "--model") { model_path = value; }
            else if (key == "--output") { output = value; }
            else if (key == "--force") { force_path = value; }
            else if (key == "--prompt") { prompt = value; }
            else if (key == "--gpu-layers") { ngl = std::stoi(value); }
            else if (key == "--ubatch") { ubatch = std::stoi(value); }
            else if (key == "--predict") { predict = std::stoi(value); }
            else if (key == "--threads") { threads = std::stoi(value); }
            else { throw std::runtime_error("unknown option: " + key); }
        }
        check(!model_path.empty() && !output.empty(), "required: --model PATH --output PREFIX");
        check(ngl >= 0 && ngl <= 999 && ubatch >= 1 && ubatch <= 32 && predict >= 1 && predict <= 32 && threads >= 1 && threads <= 12, "option outside bounded check limits");
        std::vector<llama_token> forced;
        if (!force_path.empty()) {
            std::ifstream file(force_path);
            check(bool(file), "cannot open reference metadata");
            json reference; file >> reference;
            check(reference.at("complete") == true && reference.at("prompt") == prompt, "incomplete or mismatched reference");
            forced = reference.at("evaluated_generation_ids").get<std::vector<llama_token>>();
            check(forced.size() == size_t(predict), "reference generation length mismatch");
        }
        ggml_backend_load_all();
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = ngl;
        mp.load_mode = LLAMA_LOAD_MODE_MMAP;
        const auto load_start = clock_type::now();
        auto * model = llama_model_load_from_file(model_path.c_str(), mp);
        check(model != nullptr, "model load failed");
        const double load_s = seconds(load_start);
        const auto * vocab = llama_model_get_vocab(model);
        const int nv = llama_vocab_n_tokens(vocab);
        const int nt = -llama_tokenize(vocab, prompt.data(), prompt.size(), nullptr, 0, true, true);
        check(nt > 0 && nt <= 64, "prompt must contain 1-64 tokens");
        std::vector<llama_token> tokens(nt);
        check(llama_tokenize(vocab, prompt.data(), prompt.size(), tokens.data(), nt, true, true) == nt, "tokenization failed");
        auto cp = llama_context_default_params();
        cp.n_ctx = 128; cp.n_batch = 32; cp.n_ubatch = ubatch; cp.n_seq_max = 1;
        cp.n_threads = cp.n_threads_batch = threads;
        cp.op_offload = cp.offload_kqv = ngl > 0;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.no_perf = false;
        auto * ctx = llama_init_from_model(model, cp);
        check(ctx != nullptr, "context allocation failed");
        auto * batch = llama_batch_ext_init(ctx);
        check(batch != nullptr, "batch allocation failed");
        std::ofstream logits(output + ".f32", std::ios::binary);
        check(bool(logits), "cannot open logits file");
        json report = {{"complete", false}, {"model", model_path}, {"prompt", prompt}, {"prompt_ids", tokens},
            {"gpu_layers", ngl}, {"ubatch", ubatch}, {"threads", threads}, {"n_ctx", cp.n_ctx},
            {"vocab_size", nv}, {"load_s", load_s}, {"teacher_forced", !forced.empty()}};
        std::vector<int> top_ids, evaluated;
        std::string text;
        int rows = 0;
        auto save_row = [&](int i) {
            const float * row = llama_get_logits_ith(ctx, i);
            check(row != nullptr, "missing logits");
            for (int j = 0; j < nv; ++j) { check(std::isfinite(row[j]), "nonfinite logits"); }
            logits.write(reinterpret_cast<const char *>(row), nv*sizeof(float));
            check(bool(logits), "logits write failed");
            ++rows;
            return int(std::max_element(row, row + nv) - row);
        };
        auto decode = [&](const llama_token * ids, int count, int pos) {
            llama_batch_ext_clear(batch);
            for (int i = 0; i < count; ++i) {
                const int index = llama_batch_ext_add_token(batch, 0, ids[i]);
                const llama_pos p = pos + i;
                check(index >= 0 && llama_batch_ext_set_pos(batch, index, &p), "batch position failed");
                check(llama_batch_ext_set_output_logits(batch, index, true), "batch output failed");
            }
            check(llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) == 0, "decode failed");
        };
        const auto prompt_start = clock_type::now();
        int top = -1;
        for (int first = 0; first < nt; first += ubatch) {
            const int count = std::min(ubatch, nt - first);
            decode(tokens.data() + first, count, first);
            for (int i = 0; i < count; ++i) { top = save_row(i); }
        }
        report["prompt_s"] = seconds(prompt_start);
        const auto generation_start = clock_type::now();
        // Keep evaluating a fixed number of positions, including EOG if selected.
        for (int i = 0; i < predict; ++i) {
            top_ids.push_back(top);
            const llama_token next = forced.empty() ? top : forced[i];
            check(next >= 0 && next < nv, "forced token outside vocabulary");
            evaluated.push_back(next);
            std::vector<char> piece(256);
            int n = llama_token_to_piece(vocab, next, piece.data(), piece.size(), 0, true);
            if (n < 0) { piece.resize(-n); n = llama_token_to_piece(vocab, next, piece.data(), piece.size(), 0, true); }
            check(n >= 0, "token piece conversion failed");
            text.append(piece.data(), n);
            if (i + 1 < predict) { decode(&next, 1, nt + i); top = save_row(0); }
        }
        report["generation_s"] = seconds(generation_start);
        report["prompt_tokens_s"] = nt/report["prompt_s"].get<double>();
        report["decode_tokens_s"] = (predict - 1)/report["generation_s"].get<double>();
        report["logit_rows"] = rows;
        report["greedy_ids"] = top_ids;
        report["evaluated_generation_ids"] = evaluated;
        report["text"] = text;
        logits.close();
        llama_perf_context_print(ctx);
        llama_batch_ext_free(batch);
        llama_free(ctx); llama_model_free(model); llama_backend_free();
        report["complete"] = true;
        std::ofstream metadata(output + ".json");
        metadata << report.dump(2) << '\n';
        check(bool(metadata), "metadata write failed");
        std::printf("%s\n", report.dump(2).c_str());
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "test-trellis-model: %s\n", e.what());
        return 1;
    }
}
