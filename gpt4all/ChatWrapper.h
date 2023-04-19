//
//  ChatWrapper.h
//  gpt4all
//
//  Created by Yazeed AlKhalaf on 19/04/2023.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface ChatWrapper : NSObject

- (NSString *)runChat:(NSString *)input modelPath:(NSString *)modelPath;

@end

NS_ASSUME_NONNULL_END
