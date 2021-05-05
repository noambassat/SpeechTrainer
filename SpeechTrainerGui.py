import PySimpleGUI as sg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
from matplotlib.ticker import NullFormatter
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from threading import Thread
import Audio_Processing_Backend as backend
import wave
import pyaudio


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

    for i in (range(0, int(RATE / CHUNK * RECORD_SECONDS))):
        data = stream.read(CHUNK)
        frames.append(data)
        if i % (int(RATE / CHUNK * fragmentize)) != 0 and i!=0:
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

# ===================================================================================================================#

#Layouts
Main_Screen = [
                 [sg.Button(button_text='Start Training',key='ST_BTN',button_color='red',border_width=10
                            )],
                 [sg.Button(button_text='Choose Topic',key='ST_CHOOSE_TRAIN_TOPIC')],
                 [sg.Button(button_text='Set Training Time',key='ST_CHOOSE_TRAIN_TIME')],
                 [sg.Button(button_text='Set Goals',key='ST_SET_GOALS')],
                 [sg.Button(button_text='Personal Progress and Statistics',key='ST_STATS_AND_PROG')]
               ]

Training_Screen = [
                     [sg.Button(button_text='Begin',key='T_BEGIN_REC')],
                     [sg.Text('Current Question:',justification='center',key='T_TEXT1',visible=False)],
                     [sg.Multiline('Current Suggestion: ',justification='center',key='T_Q_TEXT',visible=False)],
                     [sg.Text('What is Your Dream?:',justification='center',key='T_TEXT2',visible=False)],
                     [sg.Multiline('Raise Your Voice ',justification='center',key='T_Q_TIPS',visible=False)],
                     [sg.Button(button_text='Stop',key='T_STOP_REC',visible=False)],
                     [sg.Button(button_text='Back', key='T_BACK')],
                     [sg.Text('Audio Input:',justification='center')],
                     [sg.Canvas(key='T_SOUND_PLOT_CANVAS',visible=True)]
                      ]
# ===================================================================================================================#
#Main Layout
layout = [[sg.Column(Main_Screen, key='-COL1-'), sg.Column(Training_Screen, visible=False, key='-COL2-')]
    ]
# ===================================================================================================================#

#Window
window = sg.Window("Speech Trainer", layout,size=(800,500),auto_size_buttons=True,element_justification='center', finalize=True)
# ===================================================================================================================#
#GLOBAL VARS
_TRAIN_TIME = 0.1
_TRAIN_TOPIC = 'Null'
_TIME_COUNTER = 0
_RECORDING_STATE = False
_NEW_SEGMENT_FLAG = False
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
    print(event, values)
    if event in (None, 'Exit'):
        break

    #Main Screen Calls
# ===================================================================================================================#

    #If User Click on "Start Training"
        #Trining Calls
    if event == 'ST_BTN':
        if _TRAIN_TIME == 0 or _TRAIN_TOPIC =='null':
            if _TRAIN_TOPIC =='null':
                sg.popup_error('Please Choose Training Time and Topic First')
            else:
                sg.popup_error('Please Choose Training Time First')
            continue
        window[f'-COL1-'].update(visible=False)
        window[f'-COL2-'].update(visible=True)
    if event == 'T_BACK':
        window[f'-COL2-'].update(visible=False)
        window[f'-COL1-'].update(visible=True)
        #Begin Recording and Training process
    if event == 'T_BEGIN_REC' and _RECORDING_STATE is False:#Draw Sound Wave
        # Start Backend Recording in an new thread
        recording_thread = Thread(target=recored_audio,args=(_TRAIN_TIME * 60, 3,'','output.wav'))
        recording_thread.start()
        print('RECORDING...')
        # New Event Loop For Recording Time
        while _RECORDING_STATE:
            event, values = window.read(timeout=2)
            window['T_STOP_REC'].update(visible=True)
            window['T_Q_TEXT'].update(visible=True)
            window['T_Q_TIPS'].update(visible=True)
            window['T_TEXT1'].update(visible=True)
            window['T_TEXT2'].update(visible=True)

            #Stop Recording Event
            if event == 'T_STOP_REC':
                _RECORDING_STATE = False
                window['T_STOP_REC'].update(visible=False)
                window['T_Q_TEXT'].update(visible=False)
                window['T_Q_TIPS'].update(visible=False)
                window['T_TEXT1'].update(visible=False)
                window['T_TEXT2'].update(visible=False)
                window['T_SOUND_PLOT_CANVAS'].update(visible=False)

            #Canvas Update Section
            dpts = [np.random.randint(0, 10) for x in range(100)]
            ax.cla()
            ax.grid()
            ax.plot(range(100), dpts, color='purple')
            fig_agg.draw()

            #Segment Analysis Section
            if _NEW_SEGMENT_FLAG == True:
                print('New Segment Has Been Saved And Ready for Analysis')
                _NEW_SEGMENT_FLAG=False
            window.refresh()
        if _RECORDING_STATE == False:
            window['T_STOP_REC'].update(visible=False)
        recording_thread.join()

#===================================================================================================================#
    #Set Train Time Popup
    if event == 'ST_CHOOSE_TRAIN_TIME':
        _TRAIN_TIME = sg.popup_get_text('Choose Train Time (in Minutes)')
# ===================================================================================================================#
    #Set Train Topic Popup
    if event == 'ST_CHOOSE_TRAIN_TOPIC':
        _TRAIN_TOPIC = popup_select(['Job Interview','Date','Lecture'])
# ===================================================================================================================#

    #Visual Updates
    if _TRAIN_TIME != 0 and _TRAIN_TOPIC != 'null':
        window.FindElement('ST_BTN').Update(button_color=('green'))
window.close()
