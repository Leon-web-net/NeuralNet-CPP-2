#include "network.hpp"
#include "mnist.hpp"
#include "matrix.hpp"
#include "metrics.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace config {
    // Hyperparameters
    constexpr std::size_t EPOCHS = 25;
    constexpr std::size_t BATCH = 64;
    constexpr float       LR_START = 0.4f;
    constexpr float       LR_DECAY = 0.9f;
    constexpr std::size_t DECAY_EVERY = 5;
    constexpr float       TRAIN_FRAC = 0.8f;

    // Separate seeds: changing the shuffle never changes which images are held out.
    constexpr std::uint32_t SPLIT_SEED = 123;
    constexpr std::uint32_t SHUFFLE_SEED = 456;

    // Labels
    constexpr std::string_view BACKEND = "cpu_naive";                     // for timing logs
    constexpr std::string_view MODEL_NAME = "mlp_784-128-64-10_sigmoid_v1";  // default model file

    // Hardware peak for MFU. AMD Ryzen 7 5800X (Zen 3): 8 cores, 3.8 GHz base
    // (4.7 boost), 2 FMA units x 8 floats (AVX2) x 2 ops = 32 FP32 FLOPs/cycle/core.
    constexpr HardwarePeak CPU_PEAK{ 8.0, 3.8, 32.0 };

    // Defaults (command-line flags can switch these off)
    constexpr bool ENABLE_LOGGING = true;
    constexpr bool SAVE_MODEL = true;
}

// ---------------------------------------------------------------- options

enum class Mode { Train, TrainFull, Eval, Predict };

struct Options {
    Mode mode = Mode::Train;
    bool logging = config::ENABLE_LOGGING;
    bool save = config::SAVE_MODEL;
    std::string model_name;   // filled with a default if not given
};

constexpr std::string_view USAGE =
"usage: NeuralNet [train | train-full | eval | predict] "
"[--no-log] [--no-save] [--model NAME]";

Options parse_args(int argc, char* argv[]) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "train")       opt.mode = Mode::Train;
        else if (arg == "train-full")  opt.mode = Mode::TrainFull;
        else if (arg == "eval")        opt.mode = Mode::Eval;
        else if (arg == "predict")     opt.mode = Mode::Predict;
        else if (arg == "--no-log")    opt.logging = false;
        else if (arg == "--no-save")   opt.save = false;
        else if (arg == "--model" && i + 1 < argc) opt.model_name = argv[++i];
        else throw std::invalid_argument(
            "unknown argument '" + std::string(arg) + "'\n" + std::string(USAGE));
    }
    if (opt.model_name.empty()) {
        opt.model_name = std::string(config::MODEL_NAME);
        if (opt.mode == Mode::TrainFull) opt.model_name += "_full";
    }
    return opt;
}

std::string model_path(const std::string& name) {
    return std::string(MODELS_DIR) + "/" + name + ".bin";
}

// ---------------------------------------------------------------- helpers

struct Split {
    std::vector<std::size_t> train;
    std::vector<std::size_t> val;
};

// Same seed -> same split every time, in every mode.
Split make_split(std::size_t n, float train_frac, std::uint32_t seed) {
    std::vector<std::size_t> idx(n);
    std::iota(idx.begin(), idx.end(), std::size_t{ 0 });
    std::mt19937 gen(seed);
    std::shuffle(idx.begin(), idx.end(), gen);

    const auto n_train = static_cast<std::size_t>(train_frac * static_cast<float>(n));
    Split s;
    s.train.assign(idx.begin(), idx.begin() + n_train);
    s.val.assign(idx.begin() + n_train, idx.end());
    return s;
}

float accuracy(const Matrix& pred, const Matrix& labels) {
    if (pred.rows() != labels.rows() || pred.cols() != labels.cols()) {
        throw std::invalid_argument(
            "accuracy: pred (" + std::to_string(pred.rows()) + "x" +
            std::to_string(pred.cols()) + ") vs labels (" +
            std::to_string(labels.rows()) + "x" + std::to_string(labels.cols()) +
            ") - dimensions don't match");
    }
    const Matrix p = pred.argmax_cols();
    const Matrix t = labels.argmax_cols();
    std::size_t correct = 0;
    for (std::size_t j = 0; j < p.cols(); ++j) {
        if (p(0, j) == t(0, j)) ++correct;
    }
    return static_cast<float>(correct) / static_cast<float>(p.cols());
}

// ---------------------------------------------------------------- modes

// val_idx may be empty (train-full): validation metrics are then skipped and logged as NaN.
static void train(const mnist::Dataset& data, std::vector<std::size_t> train_idx,
    const std::vector<std::size_t>& val_idx, const Options& opt) {
    const bool has_val = !val_idx.empty();
    const Matrix X_val = data.images.gather_cols(val_idx);
    const Matrix Y_val = data.labels.gather_cols(val_idx);

    std::string run_label(config::BACKEND);
    if (!has_val) run_label += "_full";

    const double peak_gflops = config::CPU_PEAK.gflops();

    std::optional<RunLogger> logger;
    if (opt.logging) {
        const std::string run_id = make_run_id(run_label);
        logger.emplace(LOGS_DIR, run_id, peak_gflops);
        std::cout << "logging run: " << run_id << " (" << BUILD_CONFIG << ")\n";
    }
    else {
        std::cout << "logging disabled\n";
    }

    std::cout << "training on " << train_idx.size() << " images, "
        << (has_val ? "validating on " + std::to_string(val_idx.size())
            : std::string("no validation set")) << "\n";

    Network net;
    std::mt19937 gen(config::SHUFFLE_SEED);
    float lr = config::LR_START;
    const std::size_t n_train = train_idx.size();
    constexpr float NaN = std::numeric_limits<float>::quiet_NaN();

    // Work per epoch is fixed by the network shape and dataset size: count it once.
    const double epoch_flops = net.train_flops_per_sample() * static_cast<double>(n_train);
    std::cout << "work: " << net.train_flops_per_sample() / 1e3 << " kFLOP/image, "
        << epoch_flops / 1e9 << " GFLOP/epoch\n"
        << "peak: " << peak_gflops << " GFLOP/s chip ("
        << config::CPU_PEAK.gflops_per_core() << " per core)\n";

    for (std::size_t epoch = 1; epoch <= config::EPOCHS; ++epoch) {
        if (epoch % config::DECAY_EVERY == 0) lr *= config::LR_DECAY;

        std::shuffle(train_idx.begin(), train_idx.end(), gen);
        const auto t0 = std::chrono::steady_clock::now();

        float loss_sum = 0.0f;
        std::size_t n_batches = 0;

        for (std::size_t start = 0; start < n_train; start += config::BATCH) {
            const std::size_t end = std::min(start + config::BATCH, n_train);
            const std::vector<std::size_t> b_idx(train_idx.begin() + start,
                train_idx.begin() + end);
            const Matrix Xb = data.images.gather_cols(b_idx);
            const Matrix Yb = data.labels.gather_cols(b_idx);

            std::vector<Matrix> acts = net.forward(Xb);
            loss_sum += Network::cross_entropy_loss(acts.back(), Yb);
            net.update(net.backward(acts, Yb), lr);
            ++n_batches;
        }

        const auto t1 = std::chrono::steady_clock::now();
        const double secs = std::chrono::duration<double>(t1 - t0).count();
        const float train_loss = loss_sum / static_cast<float>(n_batches);
        const double images_per_sec = static_cast<double>(n_train) / secs;
        const double gflops = epoch_flops / secs / 1e9;
        const double mfu_pct = 100.0 * gflops / peak_gflops;

        float val_loss = NaN, val_acc = NaN;
        if (has_val) {
            const Matrix val_pred = net.forward(X_val).back();
            val_acc = accuracy(val_pred, Y_val);
            val_loss = Network::cross_entropy_loss(val_pred, Y_val);
        }

        if (logger) logger->log_epoch({ epoch, lr, train_loss, val_loss, val_acc, secs,
                                        images_per_sec, gflops, mfu_pct });

        std::cout << "epoch " << epoch << "  lr=" << lr << "  train_loss=" << train_loss;
        if (has_val) {
            std::cout << "  val_loss=" << val_loss << "  val_acc=" << val_acc * 100.0f << "%";
        }
        std::cout << "  (" << secs << " s, " << static_cast<long long>(images_per_sec)
            << " img/s, " << gflops << " GFLOP/s, MFU " << mfu_pct << "%)" << std::endl;
    }

    if (logger) logger->write_summary(run_label, config::BATCH, config::LR_START);

    if (opt.save) {
        std::filesystem::create_directories(MODELS_DIR);
        const std::string path = model_path(opt.model_name);
        net.save(path);
        std::cout << "saved model: " << path << "\n";
    }
}

static void evaluate(const mnist::Dataset& data, const std::vector<std::size_t>& val_idx,
    const Options& opt) {
    Network net;
    net.load(model_path(opt.model_name));

    const Matrix X = data.images.gather_cols(val_idx);
    const Matrix Y = data.labels.gather_cols(val_idx);
    const Matrix pred = net.forward(X).back();

    std::cout << "model " << opt.model_name << " on " << val_idx.size()
        << " validation images:  acc=" << accuracy(pred, Y) * 100.0f
        << "%  loss=" << Network::cross_entropy_loss(pred, Y) << "\n";
}

static void predict(const Options& opt) {
    const mnist::Dataset test =
        mnist::load(std::string(DATA_DIR) + "/test.csv", /*has_labels=*/false);

    Network net;
    net.load(model_path(opt.model_name));
    const Matrix digits = net.forward(test.images).back().argmax_cols();   // 1 x N

    std::filesystem::create_directories(SUBMISSIONS_DIR);
    const std::string path =
        std::string(SUBMISSIONS_DIR) + "/" + opt.model_name + "_submission.csv";
    std::ofstream out(path);
    if (!out) throw std::runtime_error("predict: could not open " + path);

    out << "ImageId,Label\n";
    for (std::size_t j = 0; j < digits.cols(); ++j) {
        out << (j + 1) << ',' << static_cast<int>(digits(0, j)) << '\n';   // Kaggle ids start at 1
    }
    std::cout << "wrote " << digits.cols() << " predictions to " << path << "\n";
}

// ---------------------------------------------------------------- main

int main(int argc, char* argv[]) {
    try {
        const Options opt = parse_args(argc, argv);

        if (opt.mode == Mode::Predict) {   // only needs test.csv
            predict(opt);
            return 0;
        }

        const mnist::Dataset data = mnist::load(std::string(DATA_DIR) + "/train.csv");
        const Split split = make_split(data.count, config::TRAIN_FRAC, config::SPLIT_SEED);

        switch (opt.mode) {
        case Mode::Train:
            train(data, split.train, split.val, opt);
            break;
        case Mode::TrainFull: {
            std::vector<std::size_t> all(data.count);
            std::iota(all.begin(), all.end(), std::size_t{ 0 });
            train(data, std::move(all), {}, opt);
            break;
        }
        case Mode::Eval:
            evaluate(data, split.val, opt);
            break;
        case Mode::Predict:
            break;   // handled above
        }
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
    return 0;
}