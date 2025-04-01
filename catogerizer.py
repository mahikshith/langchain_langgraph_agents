# using ollama llama3.2 model to catogerize info to different categories  

import ollama  
import os
 

model = "llama3.2" 

input_file = r"C:\Users\mahik\Documents\Git_serious\langchain_langgraph_agents\veg_cat.txt"  # input fil path 

output_file = r"C:\Users\mahik\Documents\Git_serious\langchain_langgraph_agents\veg_cat_output.txt"  # output file not yet created 

# read the input file with checking whether the file exists or not 

if os.path.exists(input_file) == False: 
    raise FileNotFoundError(f"Input file {input_file} does not exist.") 


with open(input_file, 'r') as f:
    lines = f.read().strip()  

prompt = f""" you are expert in categorizing food and grocery  items 
into different categories and if you cannot categorize the item 
then put them under 'miscellaneous' category. 
here is the list of items to categorize: 

{lines}



1. if the item is a fruit, put it under 'fruit' category or vegetable or meat or fastfood or dairy or grocery or miscellaneous etc..
2. Under each category, list the items in bullet points and sort them based on alphabetical order with in the category in an organized manner

"""

# llm response with try and except block : 

try : 
    response = ollama.generate(model = model, 
                                 prompt = prompt, 
                                 stream = False)
    
    print("After categorizing the items:")

    ans = response["response"]

    print(ans)

    print("*"*50) 

    ans1 = response.get("response","")

    print(ans1) 

    # writing the reponse to the output file 
    with open(output_file, 'w') as f: 
        f.write(ans1.strip()) 
        print(f"Response written to {output_file}")

except Exception as e:
    print(f"An error occurred: {e}")
    
