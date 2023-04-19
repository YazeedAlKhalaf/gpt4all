//
//  ChatWrapper.m
//  gpt4all
//
//  Created by Yazeed AlKhalaf on 19/04/2023.
//

#import "ChatWrapper.h"
#import "gpt4all-cpp/chat.h"
#import "gpt4all-cpp/ggml.h"

@implementation ChatWrapper

int32_t n_ctx = 2048; // context size

- (NSString *)runChat:(NSString *)input modelPath:(NSString *)modelPath {
    ggml_time_init();
//    const int64_t t_main_start_us = ggml_time_us();
    
//    int64_t t_load_us = 0;
    
    gpt_vocab vocab;
    llama_model model;
    
    // load the model
    @autoreleasepool {
//        const int64_t t_start_us = ggml_time_us();
        if (!llama_model_load(std::string([modelPath UTF8String]), model, vocab, n_ctx)) {
            NSLog(@"Failed to load the model from '%@'", modelPath);
            return @"Failed to load the model";
        }
        
//        t_load_us = ggml_time_us() - t_start_us;
    }
    
    NSLog(@"Successfully loaded the model");
    
//    int n_past = 0;
//
//    int64_t t_sample_us  = 0;
//    int64_t t_predict_us = 0;
//    
//    std::vector<float> logits;
//
//    std::vector<gpt_vocab::id> embd_inp;
//    
//    std::vector<gpt_vocab::id> instruct_inp = ::llama_tokenize(vocab, " Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n", true);
//    std::vector<gpt_vocab::id> prompt_inp = ::llama_tokenize(vocab, "### Instruction:\n\n", true);
//    std::vector<gpt_vocab::id> response_inp = ::llama_tokenize(vocab, "### Response:\n\n", false);
//    embd_inp.insert(embd_inp.end(), instruct_inp.begin(), instruct_inp.end());
//    
//    if (input.length > 0) {
//        std::vector<gpt_vocab::id> param_inp = ::llama_tokenize(vocab, std::string([input UTF8String]), true);
//        embd_inp.insert(embd_inp.end(), prompt_inp.begin(), prompt_inp.end());
//        embd_inp.insert(embd_inp.end(), param_inp.begin(), param_inp.end());
//        embd_inp.insert(embd_inp.end(), response_inp.begin(), response_inp.end());
//    }
    
    return @"implement me";
}

@end
