#!/bin/bash
# setup_mic_rv1106.sh – Reset and configure RV1106 mic for good recording

CARD=0

echo "🔄 Resetting mic input settings on card $CARD..."

# Use single-ended stereo mode (general purpose, unless you use diff wiring)
amixer -c $CARD cset name='ADC Mode' 'SingadcLR'

# Enable mic inputs and bias voltage
amixer -c $CARD cset name='ADC MIC Left' 'Work'
amixer -c $CARD cset name='ADC MIC Right' 'Work'
amixer -c $CARD cset name='ADC Main MICBIAS' 'On'

# Set mic preamp gain to max (3/3)
amixer -c $CARD cset name='ADC MIC Left Gain' 3
amixer -c $CARD cset name='ADC MIC Right Gain' 3

# Set ADC digital gain to a high but safe level
amixer -c $CARD cset name='ADC ALC Left' 20
amixer -c $CARD cset name='ADC ALC Right' 20

# Optional: disable AGC to avoid overcompensated noise
amixer -c $CARD cset name='ALC AGC Left' 'Off'
amixer -c $CARD cset name='ALC AGC Right' 'Off'

echo "✅ Mic input configured. Ready to record."

# Suggest a test
echo "👉 Test recording:"
echo "   arecord -D plughw:$CARD,0 -f S16_LE -c1 -r16000 -d 3 test.wav"
