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
int32_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());
int32_t repeat_last_n = 64; // last n tokens to penalize
int32_t n_batch = 8; // batch size for prompt processing

// sampling parameters
int32_t top_k = 40;
float top_p = 0.95f;
float temp = 0.10f;
float repeat_penalty = 1.30f;

int32_t seed = -1; // RNG seed


- (NSString *)runChat:(NSString *)input modelPath:(NSString *)modelPath {
    //    ggml_time_init();
    //    const int64_t t_main_start_us = ggml_time_us();
    
    if (seed < 0) {
        seed = time(NULL);
    }
    std::mt19937 rng(seed);
    
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
    
    int n_past = 0;
    
    //    int64_t t_sample_us  = 0;
    //    int64_t t_predict_us = 0;
    
    std::vector<float> logits;
    
    std::vector<gpt_vocab::id> embd_inp;
    
    std::vector<gpt_vocab::id> instruct_inp = ::llama_tokenize(vocab, " Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n", true);
    std::vector<gpt_vocab::id> prompt_inp = ::llama_tokenize(vocab, "### Instruction:\n\n", true);
    std::vector<gpt_vocab::id> response_inp = ::llama_tokenize(vocab, "### Response:\n\n", false);
    embd_inp.insert(embd_inp.end(), instruct_inp.begin(), instruct_inp.end());
    
//    if (input.length > 0) {
        std::vector<gpt_vocab::id> param_inp = ::llama_tokenize(vocab, std::string([input UTF8String]), true);
        embd_inp.insert(embd_inp.end(), prompt_inp.begin(), prompt_inp.end());
        embd_inp.insert(embd_inp.end(), param_inp.begin(), param_inp.end());
        embd_inp.insert(embd_inp.end(), response_inp.begin(), response_inp.end());
//    }
    
    fprintf(stderr, "sampling parameters: temp = %f, top_k = %d, top_p = %f, repeat_last_n = %i, repeat_penalty = %f\n", temp, top_k, top_p, repeat_last_n, repeat_penalty);
    fprintf(stderr, "\n\n");
    
    std::vector<gpt_vocab::id> embd;
    
    // determine the required inference memory per token:
    size_t mem_per_token = 0;
    llama_eval(model, n_threads, 0, { 0, 1, 2, 3 }, logits, mem_per_token);
    
    int last_n_size = repeat_last_n;
    std::vector<gpt_vocab::id> last_n_tokens(last_n_size);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);
    
    // we may want to slide the input window along with the context, but for now we restrict to the context length
    int remaining_tokens = model.hparams.n_ctx - embd_inp.size();
    int input_consumed = 0;
    bool input_noecho = true;
    
    while (remaining_tokens > 0) {
        // predict
        if (embd.size() > 0) {
            //        const int64_t t_start_us = ggml_time_us();
            
            if (!llama_eval(model, n_threads, n_past, embd, logits, mem_per_token)) {
                NSLog(@"Failed to predict");
                return @"Failed to predict";
            }
            
            //        t_predict_us += ggml_time_us() - t_start_us;
        }
        
        n_past += embd.size();
        embd.clear();
        
        if (embd_inp.size() <= input_consumed) {
            const int n_vocab = model.hparams.n_vocab;
            
            gpt_vocab::id id = 0;
            
            {
                //                const int64_t t_start_sample_us = ggml_time_us();
                
                id = llama_sample_top_p_top_k(vocab, logits.data() + (logits.size() - n_vocab), last_n_tokens, repeat_penalty, top_k, top_p, temp, rng);
                
                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(id);
                
                //                t_sample_us += ggml_time_us() - t_start_sample_us;
            }
            
            // add it to the context
            embd.push_back(id);
            
            // echo this to console
            input_noecho = false;
            
            // decrement remaining sampling budget
            --remaining_tokens;
        } else {
            // some user input remains from prompt or interaction, forward it to processing
            while (embd_inp.size() > input_consumed) {
                embd.push_back(embd_inp[input_consumed]);
                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(embd_inp[input_consumed]);
                ++input_consumed;
                if (embd.size() > n_batch) {
                    break;
                }
            }
        }
        
        if (!input_noecho) {
            std::string result = "";
            for (auto id : embd) {
                NSLog(@"getting thing with id: %d", id);
                result += vocab.id_to_token[id].c_str();
            }
            
            NSString *resultNS = [NSString stringWithUTF8String: result.c_str()];
            NSLog(@"result: %@", resultNS);
        }
        
        
        // end of text token
        if (embd.back() == 2) {
            printf("\n");
            fprintf(stderr, " [end of text]\n");
            break;
            
        }
        
        
    }
    
    return @"implement me";
}

@end
