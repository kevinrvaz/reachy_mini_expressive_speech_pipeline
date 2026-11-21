"""Generate matching SVG and PNG slide assets without connecting to the robot."""
from pathlib import Path
from html import escape
import math
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
SCALE = 2
im = Image.new('RGB', (1600*SCALE, 900*SCALE), '#f7f9fc')
d = ImageDraw.Draw(im)
svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900">', '<rect width="1600" height="900" fill="#f7f9fc"/>']
def rect(x,y,w,h,fill,stroke=None,r=16):
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke or fill}"/>')
    d.rounded_rectangle((x*SCALE,y*SCALE,(x+w)*SCALE,(y+h)*SCALE),radius=r*SCALE,fill=fill,outline=stroke,width=2)
def text(x,y,t,size=22,color='#192b45',bold=False):
    svg.append(f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}">{escape(t)}</text>')
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial'+(' Bold' if bold else '')+'.ttf',size*SCALE)
    d.text((x*SCALE,y*SCALE),t,font=font,fill=color,anchor='ls')
def arrow(points,color='#6b7d94',dashed=False):
    coords=' '.join(f'{x},{y}' for x,y in points)
    dash=' stroke-dasharray="7 6"' if dashed else ''
    svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="3"{dash}/>')
    d.line([(x*SCALE,y*SCALE) for x,y in points],fill=color,width=3*SCALE)
    x,y=points[-1]; a=math.atan2(y-points[-2][1],x-points[-2][0])
    triangle=[(x,y),(x-11*math.cos(a-.45),y-11*math.sin(a-.45)),(x-11*math.cos(a+.45),y-11*math.sin(a+.45))]
    svg.append('<polygon points="'+' '.join(f'{a},{b}' for a,b in triangle)+f'" fill="{color}"/>')
    d.polygon([(int(a*SCALE),int(b*SCALE)) for a,b in triangle],fill=color)
def box(x,y,w,title,lines,color='#2869c7',fill='#ffffff',h=140):
    rect(x,y,w,h,fill,'#dce4ef')
    rect(x+18,y+20,5,25,color,r=2)
    text(x+34,y+41,title,24,bold=True)
    for i,line in enumerate(lines): text(x+20,y+78+i*27,line,20)

text(60,65,'REACHY MINI  /  END-TO-END PIPELINE',18,'#2869c7',True)
text(60,119,'From spoken request to an expressive response',40,bold=True)
text(60,157,'Local voice + vision inference, concurrent speech and motion, live Rerun telemetry',23,'#61718a')
xs=[60,310,560,810,1060,1310]
for x in xs[:-1]: arrow([(x+220,285),(x+250,285)])
box(60,215,220,'Microphone',['Live audio capture','Network / WebRTC'])
box(310,215,220,'Silero VAD',['Segment speech','Detect end of turn'])
box(560,215,220,'Qwen3 ASR',['Speech → text','1.7B · MLX'])
box(810,215,220,'Address gate',['“Hey <any name>”','Gemma classifier'])
box(1060,215,220,'Think mode',['False or “low”','Gemma classifier'])
box(1310,215,220,'Gemma 4 E4B',['Stream response','Ollama'],color='#8c50ba',fill='#f4edfb')
text(60,390,'Both classifiers use current text only; no history.',19,'#61718a')
arrow([(920,355),(920,420),(680,420)],'#ba6b30')
text(335,427,'Rejected → resume listening',21,'#a45c27')
# Context enters the response model, independently of the classifiers.
box(1260,470,270,'Vision + context',['Latest camera frame','Conversation history','No frame? Text still sent.'],color='#8c50ba',h=170)
arrow([(1440,470),(1440,355)],'#8c50ba')
# Stream is parsed into two output queues.
arrow([(1350,355),(1350,444),(1000,444),(1000,490)],'#8c50ba')
box(840,490,320,'Stream parser',['<sentence> → speech queue','<emotion> → motion queue'],color='#8c50ba',h=150)
box(60,490,320,'Speech worker',['Kokoro TTS · af_heart','24 kHz → robot output rate','Audio playback on Reachy'],color='#138675',fill='#edf8f5',h=150)
box(450,490,320,'Movement worker',['Recorded emotion moves','Thinking loop while reasoning','Head, body and antennas'],color='#138675',fill='#edf8f5',h=150)
arrow([(840,565),(770,565)],'#138675')
arrow([(870,490),(870,462),(220,462),(220,490)],'#138675')
rect(60,670,1470,46,'#e8eef6',r=10)
text(80,700,'TURN CONTROL',17,'#2869c7',True)
text(260,700,'Pause VAD intake during processing and reply → wait for speech + motion → resume listening',21)
rect(60,744,1470,112,'#192b45',r=16)
text(84,781,'RERUN  ·  CONTINUOUS MONITORING',20,'#7ccef0',True)
text(84,820,'3D robot + motor angles     |     Live camera     |     Microphone waveform + RMS     |     VAD + captured speech     |     Logs',21,'#ffffff')
text(60,886,'Camera, microphone and motor telemetry run on independent capture threads. Speech and movement use separate worker queues.',18,'#61718a')
svg.append('</svg>')
(OUT/'reachy-e2e-pipeline.svg').write_text('\n'.join(svg))
im.save(OUT/'reachy-e2e-pipeline.png')
print('Created SVG (1600 × 900) and PNG (3200 × 1800).')
