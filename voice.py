
import speech_recognition as sr

while True:
    r = sr.Recognizer() 
    with sr.Microphone() as source:
    
        f = open ('result.txt','w')
        f.write("wait")
        #print ("Please wait a moment...  Calibrating microphone NOW~")
        # listen for 1 seconds and create the ambient noise energy level 
        #r.energy_threshold = 50
        r.adjust_for_ambient_noise (source, duration=1) 
        print ("Now, please say something !!!")
        res = 'tes'    
        audio = r.listen (source)
        print ("listening !!!")
    try:
        #f = open ('result.txt','w')
        print(audio)
        text = r.recognize_google(audio, language="EN")
    
        res = (text)
            #handin=res
        
        print (res)

    except sr.UnknownValueError:
        print("0")
    except sr.RequestError as e:
        print("0")
		
