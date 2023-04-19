//
//  ContentView.swift
//  gpt4all
//
//  Created by Yazeed AlKhalaf on 18/04/2023.
//

import Foundation
import SwiftUI

struct ContentView: View {
    @State private var thePrompt = "Click on me to get a random prompt"
    var body: some View {
        VStack {
            Image(systemName: "globe")
                .imageScale(.large)
                .foregroundColor(.accentColor)
            Text(thePrompt)
                .onTapGesture {
                    //                    let utils = UtilsWrapper()
                    //                    let randomPrompt = utils.randomPrompt()
                    //                    print("got random prompt: \(randomPrompt)")
                    //                    thePrompt = randomPrompt
                    
                    if let modelPath = Bundle.main.path(
                        forResource: "gpt4all-lora-quantized",
                        ofType: "bin"
                    ) {
                        print("got the model path at: \(modelPath)")
                        
                        let chatWrapper = ChatWrapper()
                        let result = chatWrapper.runChat("what are you?", modelPath: modelPath)
                        print("got result: \(result)")
                        
                        thePrompt = result
                    } else {
                        print("file not found!")
                    }
                    
                }
        }
        .padding()
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
