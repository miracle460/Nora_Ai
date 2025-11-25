
import { GoogleGenAI } from "@google/genai";
import { WebSocketServer } from 'ws';
import dotenv from 'dotenv';

// Load environment variables
dotenv.config();

// Configuration
const PORT = process.env.PORT || 8080;
const API_KEY = process.env.API_KEY;

if (!API_KEY) {
    console.error('Error: API_KEY is missing in environment variables.');
    process.exit(1);
}

const wss = new WebSocketServer({ port: PORT });
const ai = new GoogleGenAI({ apiKey: API_KEY });

console.log(`Nora Backend Server started on port ${PORT}`);
console.log(`Connect your ESP32 to ws://<YOUR_IP>:${PORT}`);

wss.on('connection', async (ws) => {
    console.log('Client connected (ESP32)');

    let session = null;

    try {
        // Establish connection to Gemini Live API
        session = await ai.live.connect({
            model: 'gemini-2.5-flash-native-audio-preview-09-2025',
            config: {
                responseModalities: ['AUDIO'], // We only need audio for the ESP32
                speechConfig: { 
                    voiceConfig: { 
                        prebuiltVoiceConfig: { voiceName: 'Zephyr' } 
                    } 
                },
                // thinkingConfig and tools removed as they are not supported on this model
                systemInstruction: "You are Nora. Your name is Nora. If asked for your name, always answer 'Nora'. Do not mention Gemini or Google. Primary Language: Malayalam. Keep responses short, concise, and friendly suitable for voice output.",
            },
            callbacks: {
                onopen: () => {
                    console.log('Connected to Gemini Live API');
                },
                onmessage: (msg) => {
                    // Forward audio chunks from Gemini to ESP32
                    const audioData = msg.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
                    if (audioData) {
                        // Gemini sends Base64, we convert to Buffer (binary) for the ESP32
                        const buffer = Buffer.from(audioData, 'base64');
                        if (ws.readyState === ws.OPEN) {
                            ws.send(buffer);
                        }
                    }
                    
                    // Log transcriptions for debugging
                    if (msg.serverContent?.outputTranscription) {
                        console.log('Nora:', msg.serverContent.outputTranscription.text);
                    }
                },
                onclose: () => {
                    console.log('Gemini Live API connection closed');
                },
                onerror: (err) => {
                    console.error('Gemini Live API Error:', err);
                }
            }
        });
    } catch (err) {
        console.error('Failed to connect to Gemini:', err);
        ws.close();
        return;
    }

    // Handle messages from ESP32
    ws.on('message', (data) => {
        // Assume ESP32 sends raw PCM data (binary)
        // We need to convert it to base64 for Gemini
        try {
            if (session) {
                // Determine if data is Buffer or ArrayBuffer
                const bufferData = Buffer.isBuffer(data) ? data : Buffer.from(data);
                const base64Audio = bufferData.toString('base64');

                session.sendRealtimeInput({
                    media: {
                        mimeType: 'audio/pcm;rate=16000', // Ensure ESP32 records at 16kHz
                        data: base64Audio
                    }
                });
            }
        } catch (err) {
            console.error('Error processing audio from ESP32:', err);
        }
    });

    ws.on('close', () => {
        console.log('Client disconnected');
        if (session) {
            session.close();
        }
    });

    ws.on('error', (error) => {
        console.error('WebSocket error:', error);
    });
});
