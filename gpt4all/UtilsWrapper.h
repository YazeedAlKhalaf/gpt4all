//
//  UtilsWrapper.h
//  gpt4all
//
//  Created by Yazeed AlKhalaf on 19/04/2023.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface UtilsWrapper : NSObject

- (NSDictionary<NSString *, NSNumber *> *)parseJSON:(NSString *)filePath;
- (NSString *)randomPrompt;
- (bool)initGPTVocabWithFile:(NSString *)fileName;
- (NSArray<NSNumber *> *)gptTokenize:(NSString *)text;
- (NSNumber *)llamaSampleTopPTopK:(NSData *)logits
                             vocab:(NSDictionary<NSString *, NSNumber *> *)_vocab
                 lastNTokens:(NSArray<NSNumber *> *)lastNTokens
                     repeatPenalty:(double)repeatPenalty
                             topK:(int)topK
                             topP:(double)topP
                             temp:(double)temp;
- (NSUInteger)ggmlQuantizeQ4_0WithSrc:(NSData *)src
                                  dst:(NSMutableData *)dst
                                    n:(int)n
                                    k:(int)k
                                   qk:(int)qk
                                 hist:(int64_t *)hist;

- (NSUInteger)ggmlQuantizeQ4_1WithSrc:(NSData *)src
                                  dst:(NSMutableData *)dst
                                    n:(int)n
                                    k:(int)k
                                   qk:(int)qk
                                 hist:(int64_t *)hist;

- (void)sampleTopKWithLogitsID:(NSMutableArray<NSDictionary *> *)logitsID topK:(int)topK;

@end

NS_ASSUME_NONNULL_END
