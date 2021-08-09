import pandas as pd 
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import numpy as np
import skfuzzy as fuzz


class Emotion_Agent(object):
    def __init__(self):
        self.Text=[]
        self.counter=0
        self.data=open('Data.txt').readlines() 
        self.features=[]
        self.labels=[]
        for i in range(len(self.data)):
            self.features.append(re.compile('".*",*').search(self.data[i]).group(0))
            self.labels.append(re.compile('";.\n').search(self.data[i]).group(0))
        
        self.processed_features = self.data_cleaner(self.features)
        self.processed_labels=self.data_cleaner(self.labels)
        
        self.Processed_data=pd.DataFrame(data=[self.processed_features,self.processed_labels]).T
        self.Processed_data.columns=['Text','Sentiment']
        self.Processed_data['Sentiment']=self.Processed_data['Sentiment'].astype(int)
        self.sampled_data=self.Processed_data.sample(frac=0.005).reset_index(drop=True) #i took 0.005 of the dataset due to memory limitations 
        self.initializer(self.sampled_data)
        
    def initializer(self,sampled_d):
        self.vectorizer = TfidfVectorizer (max_features=2000,min_df=7,max_df=0.8, stop_words=stopwords.words('english'))
        self.vectorizer.fit(sampled_d['Text'])
        self.vectorized_features=self.vectorization(sampled_d['Text'])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.vectorized_features , self.sampled_data['Sentiment'], test_size=0.2, random_state=0)
        self.model_1=self.model_1_init(self.X_train, self.X_test, self.y_train, self.y_test)
        self.model_2=self.model_2_init(self.X_train, self.X_test, self.y_train, self.y_test)
        self.model_3=self.model_3_init(self.X_train, self.X_test, self.y_train, self.y_test)
        self.model_4=self.model_4_init(self.X_train, self.X_test, self.y_train, self.y_test)
        return 
        
    # model 1 initialization (Gradient Boosting classifier)
    def model_1_init (self,X_train, X_test, y_train, y_test):
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import accuracy_score
        model_1 = GradientBoostingClassifier()
        model_1.fit(X_train, y_train)
        predictions = model_1.predict(X_test)
        from sklearn.metrics import classification_report, confusion_matrix
        print('Model 1 initialization (Gradient Boosting Classifier)')
        print(accuracy_score(y_test, predictions))
        print(confusion_matrix(y_test,predictions))
        print(classification_report(y_test,predictions))
        print('---------------------')
        return model_1
    # model 2 initialization (MLP classifier)
    def model_2_init (self,X_train, X_test, y_train, y_test):
        from sklearn.neural_network import MLPClassifier
        model_2=MLPClassifier(hidden_layer_sizes=1,max_iter=100).fit(X_train, y_train)
        model_2.score(X_test, y_test)
        print('Model 2 initialization (MLP Classifier)')
        print(model_2.score(X_test,y_test))
        print('---------------------') 
        return model_2
    # model 3 initialization (SGD Classifier)
    def model_3_init (self,X_train, X_test, y_train, y_test):
        from sklearn.linear_model import SGDClassifier
        model_3= SGDClassifier(loss='log').fit(X_train,y_train)
        print('model 3 initialization (SGD Classifier)')
        print(model_3.score(X_test,y_test))
        print('---------------------') 
        return model_3
    # model 4 initialization (Logistic Regression)
    def model_4_init (self,X_train, X_test, y_train, y_test):
        from sklearn.linear_model import LogisticRegression
        model_4=LogisticRegression(solver="lbfgs").fit(X_train,y_train)
        print('model 4 initialization (Logistic Regression)')
        print(model_4.score(X_test,y_test))
        print('---------------------')  
        return model_4

    def data_cleaner(self,Data):
        processed_features = []
        for sentence in range(0, len(Data)):
            # Remove all the special characters
            processed_feature = re.sub(r'\W', ' ', str(Data[sentence]))
            # remove all single characters
            processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)
            # Remove single characters from the start
            processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature)
            # Substituting multiple spaces with single space
            processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)
            # Removing prefixed 'b'
            processed_feature = re.sub(r'^b\s+', '', processed_feature)
            # Converting to Lowercase
            processed_feature = processed_feature.lower()
            processed_features.append(processed_feature)
        return processed_features
   
    def vectorization (self,Data):
        return self.vectorizer.transform(Data).toarray()
    
    def voter (self,C1_prob,C2_prob,C3_prob,C4_prob):
        C1_val=C1_prob
        C2_val=C2_prob
        C3_val=C3_prob
        C4_val=C4_prob
        
        #universe 
        C1 = np.arange(0,1.05,0.05)
        C2 = np.arange(0,1.05,0.05)
        C3 = np.arange(0,1.05,0.05)
        C4 = np.arange(0,1.05,0.05)
        class_result= np.arange(0,1.1,0.1)
        #Membership Functions
        C1_pos= fuzz.trimf(C1, [0.5,1,1])
        C1_ntrl= fuzz.trimf(C1, [0.5,0.5,0.5])
        C1_neg= fuzz.trimf(C1, [0,0,0.5])
        
        C2_pos= fuzz.trimf(C2, [0.5,1,1])
        C2_ntrl= fuzz.trimf(C2, [0.5,0.5,0.5])
        C2_neg= fuzz.trimf(C2, [0,0,0.5])
        
        C3_pos= fuzz.trimf(C3, [0.5,1,1])
        C3_ntrl= fuzz.trimf(C3, [0.5,0.5,0.5])
        C3_neg= fuzz.trimf(C3, [0,0,0.5])
        
        C4_pos= fuzz.trimf(C4, [0.5,1,1])
        C4_ntrl= fuzz.trimf(C4, [0.5,0.5,0.5])
        C4_neg= fuzz.trimf(C4, [0,0,0.5])
        
        class_pos= fuzz.trimf(class_result, [0.5,1,1])
        class_ntrl= fuzz.trimf(class_result,[0.5,0.5,0.5])
        class_neg= fuzz.trimf(class_result, [0,0,0.5])
        
                
        #membership functions interpretation with given values
        C1_level_neg = fuzz.interp_membership(C1, C1_neg, C1_val)
        C1_level_ntrl = fuzz.interp_membership(C1,C1_ntrl, C1_val)
        C1_level_pos = fuzz.interp_membership(C1, C1_pos, C1_val)
        
        C2_level_neg = fuzz.interp_membership(C2, C2_neg, C2_val)
        C2_level_ntrl = fuzz.interp_membership(C2,C2_ntrl, C2_val)
        C2_level_pos = fuzz.interp_membership(C2, C2_pos, C2_val)
        
        C3_level_neg = fuzz.interp_membership(C3, C3_neg, C3_val)
        C3_level_ntrl = fuzz.interp_membership(C3,C3_ntrl, C3_val)
        C3_level_pos = fuzz.interp_membership(C3, C3_pos, C3_val)
        
        C4_level_neg = fuzz.interp_membership(C4, C4_neg, C4_val)
        C4_level_ntrl = fuzz.interp_membership(C4, C4_ntrl, C4_val)
        C4_level_pos = fuzz.interp_membership(C4, C4_pos, C4_val)
        
        #Rule creation
        active_rule_neg = np.fmax(C1_level_neg, np.fmax(C2_level_neg, np.fmax(C3_level_neg,C4_level_neg)))
        Class_activation_neg = np.fmin(active_rule_neg,class_neg)
        
        active_rule_ntrl = np.fmax(C1_level_ntrl, np.fmax(C2_level_ntrl, np.fmax(C3_level_ntrl,C4_level_ntrl)))
        Class_activation_ntrl = np.fmin(active_rule_ntrl,class_ntrl)
        
        active_rule_pos = np.fmax(C1_level_pos, np.fmax(C2_level_pos, np.fmax(C3_level_pos,C4_level_pos)))
        Class_activation_pos = np.fmin(active_rule_pos, class_pos)
        
        aggregated = np.fmax(Class_activation_neg, np.fmax(Class_activation_ntrl, Class_activation_pos))
        
        # Calculate defuzzified result
        classification = fuzz.defuzz(class_result, aggregated,'centroid' )
        return classification
 
     # Action Function     
    def Analyze(self,Text,flag):
        self.cleaned_Text=self.data_cleaner(Text)
        self.vectorized_text=self.vectorization(self.cleaned_Text)
        self.C1=self.model_1.predict_proba(self.vectorized_text)
        self.C2=self.model_2.predict_proba(self.vectorized_text)
        self.C3=self.model_3.predict_proba(self.vectorized_text)
        self.C4=self.model_4.predict_proba(self.vectorized_text)
        print(Text, '\n \n')
        print('probabilities are: ',self.C1,self.C2,self.C3,self.C4)
        result=self.voter(self.C1[0][len(self.C1[0])-1],self.C2[0][len(self.C2[0])-1],self.C3[0][len(self.C3[0])-1],self.C4[0][len(self.C4[0])-1])
        print('---------------------------------')
        if result > 0.6:
            print ('The Text you gave me is ','{0:.0%}'.format(result), 'Positive. Keep up the positivity :) ')
        elif result <0.4:
            print("The text is only ","{0:.0%}".format(1-result),"Negative!. \n Why don't you cheer up a little bit! ^_^")
        else:
            print("This is confusing!. The Text you gave me is ","{0:.0%}".format(result),"Positive. \n It's neutral to me. \n Anyway, Don't lose positive Vibes")
        
        #update the Knowledge Base to increase efficiency, actual classes are taken as input 
        while flag:
            self.Actual_class= input('what is the actual sentiment of the text? (P: Positive, U: Neutral, N: Negative)').lower()
            if self.Actual_class not in ['p','n','u']:
                print('Wrong Input, Try Again!')
                continue
            elif  self.sampled_data['Text'].eq(str(self.cleaned_Text)).sum():
                break
            elif self.Actual_class =='p':
                self.Actual_class=4
                self.to_append=pd.DataFrame(data={'Sentiment': int(self.Actual_class),'Text':str(self.cleaned_Text)},index=['0']) 
                self.sampled_data=self.sampled_data.append(self.to_append,ignore_index=True)
                self.counter+=1
                break
            elif self.Actual_class == 'u':
                self.Actual_class=2
                self.to_append=pd.DataFrame(data={'Sentiment': int(self.Actual_class),'Text':str(self.cleaned_Text)},index=['0']) 
                self.sampled_data=self.sampled_data.append(self.to_append,ignore_index=True)
                self.counter+=1
                break
            elif self.Actual_class == 'n':
                self.Actual_class=0
                self.to_append=pd.DataFrame(data={'Sentiment': int(self.Actual_class),'Text':str(self.cleaned_Text)},index=['0']) 
                self.sampled_data=self.sampled_data.append(self.to_append,ignore_index=True)
                self.counter+=1
                break
            
        if flag ==0:
            
            if result <0.4:
                self.Act_class=0
            elif result >0.6:
                self.Act_class=4
            else: self.Act_class=2
            self.to_append=pd.DataFrame(data={'Sentiment': int(self.Act_class),'Text':str(self.cleaned_Text)},index=['0']) 
            self.sampled_data=self.sampled_data.append(self.to_append,ignore_index=True)
            self.counter+=1
            
        if self.counter == 500 :    #Re-train the model when there are 500 new samples
            print('New samples are being learned, Kindly wait... ')
            self.sampled_data.drop(labels=np.arange(0,500,1),inplace=True) # to keep running performance the same, can be ommitted
            self.initializer(self.sampled_data)
            self.counter=0
            print('Learning finished, back online!')
                             
                    
        return result



'----------Main----------'
Agent=Emotion_Agent()

while 1:
    Text=[input('Enter the text to analyze: \n')]
    Agent.Analyze(Text,1)


data=pd.read_csv('reviews.csv')
data['sentiment']=None
for i in range(len(data)):
    t=[str(data.iloc[i]['text'])]
    data.at[i,'sentiment']=Agent.Analyze(t,0)
    if data.at[i,'sentiment'] <0.4:
        data.at[i,'sentiment']=0
    elif data.at[i,'sentiment'] >0.6:
        data.at[i,'sentiment']=4
    else: data.at[i,'sentiment']=2
    
data['validation']=(data['rating']==data['sentiment'])
val=data['validation'].value_counts()
print(val)
print('system is correct in predictions: ',"{0:.0%}".format(val[1]/(val.sum())),' of the times')
