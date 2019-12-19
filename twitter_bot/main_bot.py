from nlp_helper import preprocessing
from data_processor import AskeBaytwitter
from nlp_helper import clean_at
from MyBot import AskeBayBot

if __name__ == "__main__":
    
#    print("I'm a main")
    # Add the sentiment detection to chatbot
    askeBay = AskeBaytwitter("data", "twcs.csv.zip")
    askeBay.read_data()
    questions, responses = askeBay.get_data()

    # ask a Bot
    ebayBot = AskeBayBot(questions ,responses)
    exit_codes = ['bye', 'see you', 'c ya', 'exit']
    flag=True
    print("Hi! Im an ebayBot, I will try to answer your queries !")

    while(flag):
        user_response = input("User:")
        if user_response.lower() not in exit_codes:
            
            user_response = user_response.lower()
            sim_query, sim_answer, sim_score = ebayBot.get_response(user_response)
            print("eBayBot :", sim_answer)
            print('\nDo you want to continue ? (yes/no)')
            user_response = input("User-:yes/no? ")
            if user_response.lower() == 'no' or user_response.lower() in exit_codes :
                    print('Bye!!')
                    flag=False

        else :
                print('Bye!!')
                flag=False

    '''
    #query = "I need an assistance. The item is marked delivery but I never receive it"
    query = "I need to contact customer service to report an purchase issue"
    
    sim_query, sim_answer, sim_score = ebayBot.get_response(query)
    print("The possible response: {}".format(sim_answer))
    print("The similar question: {}".format(sim_query))

    sim_queries, sim_answers, sim_scores = ebayBot.get_all_responses(query)
    print("All possible responses: ")
    [print(resp) for resp in sim_answers]
    
    '''
    

