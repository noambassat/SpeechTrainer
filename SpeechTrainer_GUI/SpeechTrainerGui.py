import PySimpleGUI as sg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from matplotlib.figure import Figure
from threading import Thread
from SpeechTrainer_GUI import Audio_Processing_Backend as backend
import wave
import pyaudio
import json
import speech_recognition as sr
import os
import nltk
from nltk import FreqDist
from nltk.corpus import brown

sg.theme('LightBrown6')


#FUNCS
#Popup select window
def popup_select(the_list,select_multiple=False):
    layout = [[sg.Listbox(the_list,key='_LIST_',size=(45,len(the_list)),select_mode='extended' if select_multiple else 'single',bind_return_key=True),sg.OK()]]
    window = sg.Window('Select One',layout=layout)
    event, values = window.read()
    window.close()
    del window
    if select_multiple or values['_LIST_'] is None:
        return values['_LIST_']
    else:
        return values['_LIST_'][0]
#plt helper for canvas creation
def draw_figure(canvas, figure, loc=(0, 0)):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg



def recored_audio(rec_time  , fragmentize=2, output_path='', file_name='output.wav'):
    """
    Parameters
    ----------
    rec_time : int
        The number of secondes the function will recored
    output_path : str
        where to save the recording
    file_name : str
        the name of the saved file followed by .wav
    fragmentize : int
        every x seconds save a fragment.wav file of the last x seconds in the recording
    """

    # =====================WAV Audio Format Static Variables ============================
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = rec_time
    WAVE_OUTPUT_FILENAME = output_path + file_name
    global _RECORDING_STATE
    global _NEW_SEGMENT_FLAG
    _RECORDING_STATE = True
    p = pyaudio.PyAudio()



    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []
    segment = []
    global _STOP_RECORDING
    for i in (range(0, int(RATE / CHUNK * RECORD_SECONDS))):
        if _STOP_RECORDING == True:
            break
        data = stream.read(CHUNK)
        frames.append(data)
        if i % (int(RATE / CHUNK * fragmentize)) != 0 or i == 0:
            segment.append(data)
        else:
            print('# fragment saved')
            wf = wave.open(output_path + 'fragment.wav', 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(segment))
            wf.close()
            _NEW_SEGMENT_FLAG = True
            segment = []

    _RECORDING_STATE = False
    print("* done recording",_RECORDING_STATE)

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def calculate_sound_features(snd):
        # Calculate A Bulck of prrat features simmilar to what we saw in MIT's paper
        pitch = snd.to_pitch().selected_array['frequency']
        intensity = snd.to_intensity().values
        max_pitch = np.max(pitch)
        min_pitch = np.min(pitch)
        mean_pitch = np.mean(pitch)
        diffPitchMaxMean = max_pitch - mean_pitch
        intensityMean = np.mean(intensity)
        intensityMax = np.max(intensity)
        intensityMin = np.min(intensity)
        broad_pitch = backend.call(snd, "To Pitch", 0.0, 50, 600)  # create a praat pitch object
        minF0 = backend.call(broad_pitch, "Get minimum", 0, 0, "hertz", "Parabolic")  # get min pitch
        maxF0 = backend.call(broad_pitch, "Get maximum", 0, 0, "hertz", "Parabolic")  # get max pitch
        floor = minF0 * 0.9
        ceiling = maxF0 * 1.1
        pitch = backend.call(snd, "To Pitch", 0.0, floor, ceiling)  # create a praat pitch object
        duration = backend.call(snd, "Get total duration")  # duration
        meanF0 = backend.call(pitch, "Get mean", 0, 0, "hertz")  # get mean pitch
        stdevF0 = backend.call(pitch, "Get standard deviation", 0, 0, "hertz")  # get standard deviation
        harmonicity = backend.call(snd, "To Harmonicity (cc)", 0.01, minF0, 0.1, 1.0)
        hnr = backend.call(harmonicity, "Get mean", 0, 0)
        pointProcess = backend.call(snd, "To PointProcess (periodic, cc)", minF0, maxF0)
        localJitter = backend.call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        if meanF0 > 170 and meanF0 < 300:
            max_formant = 5500
        elif meanF0 <= 170:
            max_formant = 5000
        elif meanF0 >= 300:
            max_formant = 8000
        else:
            max_formant = 9000

        formants = backend.call(snd, "To Formant (burg)", 0.0025, 5, max_formant, 0.025, 50)
        numPoints = backend.call(pointProcess, "Get number of points")
        f1_list = []
        f2_list = []
        f3_list = []
        f4_list = []
        for point in range(0, numPoints):
            point += 1
            t =  backend.call(pointProcess, "Get time from index", point)
            f1 = backend.call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
            f2 = backend.call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
            f3 = backend.call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
            f4 = backend.call(formants, "Get value at time", 4, t, 'Hertz', 'Linear')
            f1_list.append(f1)
            f2_list.append(f2)
            f3_list.append(f3)
            f4_list.append(f4)
            f1_list.append(f1)
            f2_list.append(f2)
            f3_list.append(f3)
            f4_list.append(f4)

        # return relvent to us feature *delete later those that have no use
        global _ANALYSIS_RESULT
        global _ANALYSIS_RESULT_FLAG
        _ANALYSIS_RESULT = {'intensityMax': intensityMax,
                       'intensityMin': intensityMin,
                       'intensityMean': intensityMean,
                       'PitchMax': max_pitch,
                       'PitchMin': min_pitch,
                       'PitchMean': mean_pitch,
                       'diffPitchMaxMean': max_pitch-mean_pitch,
                       'F1_STD': np.std(f1_list),
                       'F3_STD': np.std(f3_list),
                       'F2_STD_F1': np.std(f2_list) / np.std(f1_list),
                       'pitch':pitch
                        }
        _ANALYSIS_RESULT_FLAG = True


# ===================================================================================================================#

#Layouts
Main_Screen = [
                 [sg.Button(button_text='Start Training',key='ST_BTN',button_color='red',border_width=10,pad=(1,1))],
                 [sg.Button(button_text='Choose Topic',key='ST_CHOOSE_TRAIN_TOPIC')],
                 [sg.Button(button_text='Set Training Time',key='ST_CHOOSE_TRAIN_TIME')],
                 [sg.Button(button_text='Set Goals',key='ST_SET_GOALS')],
                 [sg.Button(button_text='Personal Progress and Statistics',key='ST_STATS_AND_PROG')]
               ]


QS_Frame_Layout = [
                     [sg.Text('Current Question:', justification='center', key='T_TEXT1', visible=False)],
                     [sg.Multiline('Current Question: ', justification='center', key='T_Q_TEXT', visible=False),
                      sg.Button(button_text='Next Question', key='T_NEXT_Q', visible=False)],
                     [sg.Text('Current Suggestion:', justification='center', key='T_TEXT2', visible=False)],
                     [sg.Multiline('Raise Your Voice ', justification='center', key='T_Q_TIPS', visible=False)],

                  ]
Training_Screen = [
                     [sg.Button(button_text='Begin',key='T_BEGIN_REC')],
                     [sg.Frame('Status Panel', QS_Frame_Layout, font='Any 12', title_color='blue'),
                      sg.Canvas(key='T_PROGRESS_PLOT_CANVAS', visible=True)],
                     [sg.Button(button_text='Stop',key='T_STOP_REC',visible=False)],
                     [sg.Button(button_text='Back', key='T_BACK')],
                    [sg.Button(button_text='Calculate Text', key='Cal_Text',visible=False)],
                     [sg.Text('Audio Input:',justification='center')],
                     [sg.Canvas(key='T_SOUND_PLOT_CANVAS',visible=True)]
                      ]
Full_Text_Layout =[
                [sg.Text('Full Text', justification='center', key='text_print', visible=True)],
                [sg.Multiline( justification='center', key='print_full_text', visible=True)]
                ]

Words_freq_Layout = [
                    [sg.Text('Words and frequency', justification='center', key='WORD_FREQ', visible=True)],
                     [sg.Multiline('Words and frequency: ', justification='center', key='F_WORDS_LINES', visible=True),
                      sg.Button(button_text='Next word', key='C_next_word', visible=False)]
                    ]

Text_Calculating_Screen = [
                    [sg.Button(button_text='Back', key='C_BACK')],
                    [sg.Button(button_text='Print text', key='C_PRINT_TEXT')],
                    [sg.Frame('Words Frequency', Words_freq_Layout, font='Any 12', title_color='blue'),
                        sg.Canvas(key='PRINT_WORDS_FREQ', visible=True)],
                    [sg.Frame('Print text', Full_Text_Layout, font='Any 12', title_color='blue'),
                     sg.Canvas(key='PRINT_TEXT', visible=False)]
                    ]
# ===================================================================================================================#
#Main Layout
layout = [[sg.Column(Main_Screen, key='-COL1-'), sg.Column(Training_Screen, visible=False, key='-COL2-'), sg.Column(Text_Calculating_Screen, visible=False, key='-COL3-')]
    ]
# ===================================================================================================================#

#Window
window = sg.Window("Speech Trainer", layout,size=(800,500),auto_size_buttons=True,element_justification='center', finalize=True)
# ===================================================================================================================#

#Loading External Data
with open('/Users/Noam/PycharmProjects/NLP/SpeechTrainer_GUI/Data/Job_Interview_Questions.json','r') as jfile:
    Q_Job_Interview = json.load(jfile)
    Q_Job_Interview = json.loads(Q_Job_Interview)
Q_Job_Interview = [i for i in list(Q_Job_Interview.keys()) if i.find('?')!=-1]
with open('/Users/Noam/PycharmProjects/NLP/SpeechTrainer_GUI/Data/Date_Questions.json','r') as jfile:
    Q_Date = json.load(jfile)
    Q_Date = json.loads(Q_Date)


# ===================================================================================================================#
#GLOBAL VARS
_STARTING_SCORE       = 0.5
_TRAIN_TIME           = 1
_TRAIN_TOPIC          = 'Null'
_TIME_COUNTER         = 0
_SAMPLING_RATE        = 2
_NEXT_WORD_ = 0
_RECORDING_STATE      = False
_NEW_SEGMENT_FLAG     = False
_ANALYSIS_RESULT_FLAG = False
_ANALYSIS_RESULT      = None
_AX_TAG               = None
_AX_TAG_2             = None
_QUESTION_SETS        = {
                          'Job Interview':Q_Job_Interview,
                          'Date':Q_Date,
                          'Lecture':Q_Job_Interview
                        }
_LOADED_QUESTION_SET = None
_STOP_RECORDING      = False
# ===================================================================================================================#
#Canvas Config
canvas_elem = window['T_SOUND_PLOT_CANVAS']
canvas = canvas_elem.TKCanvas

fig = Figure(figsize=(9,4))
ax = fig.add_subplot(111)
ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
ax.grid()
fig_agg = draw_figure(canvas, fig)


# ===================================================================================================================#
while True:
    event, values = window.read()
    if event in (None, 'Exit'):
        print('EXIT')
        os.remove('fragment.wav')
        # os.remove('output.wav')
        break

    #Main Screen Calls
# ===================================================================================================================#

    #If User Click on "Start Training"
        #Trining Calls
    if event == 'ST_BTN':
        if _TRAIN_TIME == 0 or _TRAIN_TOPIC =='Null':
            if _TRAIN_TOPIC =='Null':
                sg.popup_error('Please Choose Training Topic First')
            else:
                sg.popup_error('Please Choose Training Time First');
            continue
        window[f'-COL1-'].update(visible=False)
        window[f'-COL2-'].update(visible=True)

    if event == 'T_BACK':
        window[f'-COL2-'].update(visible=False)
        window[f'-COL1-'].update(visible=True)

        # window[f'-COL2-'].update(visible=False)
        # window[f'-COL1-'].update(visible=True)

        #Begin Recording and Training process
    if event == 'T_BEGIN_REC' and _RECORDING_STATE is False:#Draw Sound Wave
        # Start Backend Recording in an new thread
        recording_thread = Thread(target=recored_audio,args=(_TRAIN_TIME * 60, _SAMPLING_RATE,'','output.wav'))
        recording_thread.start()
        print('RECORDING...')
        #Load First Question
        window['T_Q_TEXT'].update(_LOADED_QUESTION_SET[np.random.randint(0,len(_LOADED_QUESTION_SET),1)[0]])


        # New Event Loop For Recording Time
        while _RECORDING_STATE:
            event, values = window.read(timeout=2)
            window['T_STOP_REC'].update(visible=True)
            window['T_Q_TEXT'].update(visible=True)
            window['T_Q_TIPS'].update(visible=True)
            window['T_TEXT1'].update(visible=True)
            window['T_TEXT2'].update(visible=True)
            window['T_NEXT_Q'].update(visible=True)
            window['T_BACK'].update(visible=False)
            window['T_BEGIN_REC'].update(visible=False)

            #Stop Recording Event
            if event == 'T_STOP_REC':
                _RECORDING_STATE = False
                window['T_STOP_REC'].update(visible=False)
                window['T_BEGIN_REC'].update(visible=True)
                window['T_Q_TEXT'].update(visible=False)
                window['T_Q_TIPS'].update(visible=False)
                window['T_TEXT1'].update(visible=False)
                window['T_TEXT2'].update(visible=False)
                window['T_NEXT_Q'].update(visible=False)
                window['T_SOUND_PLOT_CANVAS'].update(visible=False)
                window['T_BACK'].update(visible=True)

            #Segment Analysis Section
            if _NEW_SEGMENT_FLAG == True:
                print('New Segment Has Been Saved And Ready for Analysis')
                print('Starting Analysis in new Thread')
                segment = backend.pm.Sound('fragment.wav')
                analysis_thread = Thread(target=calculate_sound_features,args=[segment])
                analysis_thread.start()
                _NEW_SEGMENT_FLAG=False
            #Analysis Thread Has Finished Calculating Features Training Screen Update Acording To Result
            if _ANALYSIS_RESULT_FLAG == True:
                _ANALYSIS_RESULT_FLAG=False
                analysis_thread.join()
                print('Starting Window Update Based on Analysis Results')

                #Updating
                print(_ANALYSIS_RESULT)
                cur_suggestion,cur_score = backend.get_score_and_suggestion(_ANALYSIS_RESULT,_STARTING_SCORE)
                _STARTING_SCORE = cur_score
                window['T_Q_TIPS'].update(cur_suggestion+' \t'+'Current Speech Score: [ %0.2f ]'%_STARTING_SCORE)

                ####

                # Canvas Update Section
                ax.cla()
                #in case there are already 2 layers on canvas
                if _AX_TAG != None:
                    _AX_TAG.cla()
                    _AX_TAG_2.cla()
                _AX_TAG,_AX_TAG_2 = backend.draw_gui_soundgraph(_ANALYSIS_RESULT['pitch'],segment,ax)
                fig_agg.draw()
                # Next Question Button Pressed
            if event == 'T_NEXT_Q':
                window['T_Q_TEXT'].update(_LOADED_QUESTION_SET[np.random.randint(0, len(_LOADED_QUESTION_SET), 1)[0]])

            if event == 'T_STOP_REC':
                _STOP_RECORDING=True
                recording_thread.join()
                window['Cal_Text'].update(visible=True)
                break
            window.refresh()

        if _RECORDING_STATE == False:
            window['T_STOP_REC'].update(visible=False)
            _STOP_RECORDING=False

    if event == 'Cal_Text':
        window[f'-COL2-'].update(visible=False)
        window[f'-COL3-'].update(visible=True)
        window['C_next_word'].update(visible=True)
        print('calculating...')
        r = sr.Recognizer()
        with sr.AudioFile('output.wav') as source:
            audio = r.record(source)
            words_list = []
            try:
                text = r.recognize_google(audio)
                freq = FreqDist(text.split())
                ### note: We can calculate if there are too many words repititions and present them
                for word, freq in freq.items():
                    print_freq = "The word: '" + word + "' has been said " + str(freq) + " times."
                    words_list.append(print_freq)
            except:
                print("couldn't recognize text")

    if event == 'C_BACK':
        _NEXT_WORD_ = 0
        window[f'-COL3-'].update(visible=False)
        window[f'-COL2-'].update(visible=False)
        window[f'-COL1-'].update(visible=True)

        #Join Recording Thread To Main Process
        recording_thread.join()

    if event == 'C_next_word':
        window['F_WORDS_LINES'].update(words_list[_NEXT_WORD_])
        _NEXT_WORD_ += 1
    if event == 'C_PRINT_TEXT':
        window['print_full_text'].update(text)
    #===================================================================================================================#
    #Set Train Time Popup
    if event == 'ST_CHOOSE_TRAIN_TIME':
        _TRAIN_TIME = sg.popup_get_text('Choose Train Time (in Minutes)')
# ===================================================================================================================#
    #Set Train Topic Popup
    if event == 'ST_CHOOSE_TRAIN_TOPIC':
        _TRAIN_TOPIC = popup_select(['Job Interview','Date','Lecture'])
        _LOADED_QUESTION_SET = _QUESTION_SETS[_TRAIN_TOPIC]


# ===================================================================================================================#

    #Visual Updates
    if _TRAIN_TIME != 0 and _TRAIN_TOPIC != 'null':
        window.FindElement('ST_BTN').Update(button_color=('green'))
window.close()
