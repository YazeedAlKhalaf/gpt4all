//
//  UtilsWrapper.m
//  gpt4all
//
//  Created by Yazeed AlKhalaf on 19/04/2023.
//

#import "UtilsWrapper.h"
#import "gpt4all-cpp/utils.h"

@implementation UtilsWrapper

- (NSDictionary<NSString *, NSNumber *> *)parseJSON:(NSString *)filePath {
    std::map<std::string, int32_t> resultMap = json_parse(filePath.UTF8String);
    NSMutableDictionary<NSString *, NSNumber *> *jsonResult = [NSMutableDictionary new];

    for (auto &pair : resultMap) {
        NSString *key = [NSString stringWithUTF8String:pair.first.c_str()];
        jsonResult[key] = @(pair.second);
    }
    return jsonResult;
}

- (NSString *)randomPrompt {
    std::mt19937 rng;
    std::string result = gpt_random_prompt(rng);
    return [NSString stringWithUTF8String:result.c_str()];
}

- (bool)initGPTVocabWithFile:(NSString *)fileName {
    gpt_vocab vocab;
    return gpt_vocab_init(fileName.UTF8String, vocab);
}

- (NSArray<NSNumber *> *)gptTokenize:(NSString *)text {
    gpt_vocab vocab;
    std::vector<gpt_vocab::id> tokenized = gpt_tokenize(vocab, text.UTF8String);
    NSMutableArray<NSNumber *> *result = [NSMutableArray new];
    
    for (const auto &id : tokenized) {
        [result addObject:@(id)];
    }
    return result;
}

- (NSNumber *)llamaSampleTopPTopK:(NSData *)logits
                             vocab:(NSDictionary<NSString *, NSNumber *> *)_vocab
                 lastNTokens:(NSArray<NSNumber *> *)lastNTokens
                     repeatPenalty:(double)repeatPenalty
                             topK:(int)topK
                             topP:(double)topP
                             temp:(double)temp {
    gpt_vocab vocab;
    const float *logitsPtr = (const float *)logits.bytes;

    std::vector<gpt_vocab::id> _lastNTokens;
    for (NSNumber *tokenID in lastNTokens) {
        _lastNTokens.push_back((gpt_vocab::id)tokenID.intValue);
    }

    std::mt19937 rng;
    gpt_vocab::id sampledID = llama_sample_top_p_top_k(vocab, logitsPtr, _lastNTokens,
                                                        repeatPenalty, topK, topP, temp, rng);
    return @(sampledID);
}

- (NSUInteger)ggmlQuantizeQ4_0WithSrc:(NSData *)src
                                  dst:(NSMutableData *)dst
                                    n:(int)n
                                    k:(int)k
                                   qk:(int)qk
                                 hist:(int64_t *)hist {
    return ggml_quantize_q4_0((float *)src.bytes, (void *)dst.mutableBytes, n, k, qk, hist);
}

- (NSUInteger)ggmlQuantizeQ4_1WithSrc:(NSData *)src
                                  dst:(NSMutableData *)dst
                                    n:(int)n
                                    k:(int)k
                                   qk:(int)qk
                                 hist:(int64_t *)hist {
    return ggml_quantize_q4_1((float *)src.bytes, (void *)dst.mutableBytes, n, k, qk, hist);
}

- (void)sampleTopKWithLogitsID:(NSMutableArray<NSDictionary *> *)logitsID topK:(int)topK {
    std::vector<std::pair<double, gpt_vocab::id>> _logitsID;
    
    for (NSDictionary *pair in logitsID) {
        NSNumber *probability = pair[@"probability"];
        NSNumber *id = pair[@"id"];
        _logitsID.push_back(std::make_pair(probability.doubleValue, id.intValue));
    }
    
    sample_top_k(_logitsID, topK);
    
    [logitsID removeAllObjects];
    for (const auto &pair : _logitsID) {
        NSDictionary *newPair = @{@"probability": @(pair.first), @"id": @(pair.second)};
        [logitsID addObject:newPair];
    }
}

@end
