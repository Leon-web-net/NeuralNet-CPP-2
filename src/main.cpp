#include "cli.hpp"
#include "commands.hpp"
#include "dataset_split.hpp"
#include "mnist.hpp"
#include <exception>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    try {
        const Options opt = parse_args(argc, argv);

        if (opt.mode == Mode::Predict) {   // only needs test.csv
            run_predict(opt);
            return 0;
        }

        const mnist::Dataset data = mnist::load(std::string(DATA_DIR) + "/train.csv");
        const Split split = make_split(data.count, opt.cfg.train_frac, opt.cfg.split_seed);

        switch (opt.mode) {
        case Mode::Train:     run_train(data, split.train, split.val, opt);        break;
        case Mode::TrainFull: run_train(data, all_indices(data.count), {}, opt);   break;
        case Mode::Eval:      run_eval(data, split.val, opt);                      break;
        case Mode::Predict:   break;   // handled above
        }
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
