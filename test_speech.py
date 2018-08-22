import speech_recognition as sr
import webbrowser as wb
import time

# create a txt format document and write in the result after execution

# setting the position of chrome.exe
#chrome_path = "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s"


# obtain audio from the microphone
r = sr.Recognizer() 
def mainlisten():
    while True:
        

        with sr.Microphone() as source:
            f = open ('result.txt','w')
            f.write("wait")
            print ("Please wait a moment...  Calibrating microphone NOW~")
            # listen for 1 seconds and create the ambient noise energy level 
            r.adjust_for_ambient_noise (source, duration=5) 
            f = open ('result.txt','w')
            f.write("voice")
            print ("Now, please say something !!!")
            
            audio = r.listen (source)
    
    # recognize speech using Google Cloud Speech API and start browser searching
        try:
            f = open ('result.txt','w')
            text = r.recognize_google(audio, language="EN")
            if text == 'close':
                break

            print ("Google Cloud Speech API thinks you said :\n" + text)
            
            print (r.recognize_google(audio, language="EN"), file = f)
            #f_text =  text + ".com"
            f_text =  text
            
            f.write(f_text)
            
            #wb.get (chrome_path) .open (f_text)
            #wb.open('http://127.0.0.1:5000/' + f_text)
            print ("What you said has been saved as [result.txt] :)")
        except sr.UnknownValueError:
            f = open ('result.txt','w')
            f.write("wait")
            print ("Google Cloud Speech API could not understand audio, please retry again...")
        except sr.RequestError as e:
            f = open ('result.txt','w')
            f.write("wait")
            print ("No response from Google Cloud Speech API service: {0}, please retry later... :(".format(e))
        f.close()
        time.sleep(5)
# show the result written in txt 5 seconds
#f = open ('result.txt','r')
#print (f.read())

mainlisten()

    