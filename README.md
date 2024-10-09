# ThematicApperceptionTest

>> ***This page summarizes the content of a paper presented at a conference in November 2018..***

## Summary
Most psychological test applications in Korea are simple multiple-choice tests, focused on career or personality types for adults. This study developed a projective psychological test system that analyzes descriptive sentences provided by users using natural language processing (NLP). The system stores test responses in a database and uses Google's NLP technology to analyze and present results to the therapist, offering objective analysis and reducing interpretation time.


### Psychological Test System Design
The overall system structure is shown in Figure 1. The pre-processing phase involves saving the user’s input into the database after the web-based test, while the post-processing phase uses Google's NLP API to analyze the input data stored in the database.

Figure 1) System Flow Diagram
3.1 Pre-processing phase

<div>
 
<img width = "500" src ="https://user-images.githubusercontent.com/28712478/52125197-0c3c1400-266f-11e9-951c-293d20b0245e.png">

</div>


<div>
<img width="500" src = "https://user-images.githubusercontent.com/28712478/52125654-7d2ffb80-2670-11e9-9a56-ff419f30daa6.png">
 
 
 (Figure 2) Examination Page  
</div>
   
When users enter the web page, they begin by selecting their gender, then proceed with the test. Each test page resembles Figure 2, where users create their own story based on an image and input it into a text area. Before moving to the next page, the system checks if the text has been entered. If not, it alerts the user to input text and prevents page navigation. If text is entered, it is saved into the stringDB, and the time taken from the start of the first question until moving to the second is recorded using the time package and stored in the time DB. The therapist can later review how long the user took to respond to each question and identify specific questions where responses took longer.





### 3.2 Post-processing Phase
To analyze the data stored in the database, the system used the Word2Vec function and Google's AI API. Word2Vec vectors words based on their meanings and measures similarity between words, extracting words related to the tester's questions. However, because the input text from users was often too short, and Korean language structures differ from English (with particles attached to nouns, like "여자는" versus "여자가"), Word2Vec was not effective for this project. As a result, the Google NLP API was used instead.

Among the methods provided by Google’s NLP API, entity sentiment analysis and content classification are not supported for Korean, so only the sentiment analysis function was used. This analysis assesses whether the overall sentiment in the text is positive or negative. Sentiment is expressed as a score ranging from -1.0 (negative) to 1.0 (positive), while magnitude represents the intensity of the sentiment, with higher values indicating stronger emotions.



### 3.3 Implementation Results


<img width = "600" src = "https://user-images.githubusercontent.com/28712478/52125910-3ee70c00-2671-11e9-8b7b-8e98f69c1ccc.png">
(Table 1)  Analysis Results
The system returns sentiment scores for each question. For example, in question 1, the overall sentiment score is -0.4, indicating negative content, while question 2 shows a more peaceful and positive sentiment with a score of 0.69. On a sentence level, the most positive sentence in question 2 describes "beautiful piano melodies" and "the image of a beautiful son," while a more factual statement scores lower.
