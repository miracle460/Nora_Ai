
import React, { useState, useRef, useCallback, useEffect } from 'react';
import { GoogleGenAI, LiveServerMessage, Modality, Blob as GenAiBlob } from '@google/genai';
import { encode, decode, decodeAudioData } from '../utils/audioUtils';

/**
 * Robustly downsamples (or upsamples) audio data to 16kHz for the Gemini API.
 * Handles various input sample rates (44.1k, 48k, 96k, etc.) to prevent "Internal error encountered".
 */
const createPcmBlobForGemini = (inputData: Float32Array, inputSampleRate: number): GenAiBlob => {
    const targetSampleRate = 16000;
    
    // If rates match, pass through (rare but possible)
    if (inputSampleRate === targetSampleRate) {
         const l = inputData.length;
         const int16 = new Int16Array(l);
         for (let i = 0; i < l; i++) {
             const s = Math.max(-1, Math.min(1, inputData[i]));
             int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
         }
         return {
            data: encode(new Uint8Array(int16.buffer)),
            mimeType: `audio/pcm;rate=${targetSampleRate}`,
        };
    }

    // Calculate resampling ratio
    const ratio = inputSampleRate / targetSampleRate;
    const newLength = Math.floor(inputData.length / ratio);
    const finalData = new Float32Array(newLength);
    
    // Linear Interpolation Resampling
    for (let i = 0; i < newLength; i++) {
        const offset = i * ratio;
        const index1 = Math.floor(offset);
        const index2 = Math.min(index1 + 1, inputData.length - 1);
        const weight = offset - index1;
        
        // Bounds check
        const val1 = inputData[index1] || 0;
        const val2 = inputData[index2] || 0;
        
        finalData[i] = val1 * (1 - weight) + val2 * weight;
    }

    // Convert to Int16 PCM
    const l = finalData.length;
    const int16 = new Int16Array(l);
    for (let i = 0; i < l; i++) {
        // Clamp values to [-1, 1] then scale to 16-bit range
        const s = Math.max(-1, Math.min(1, finalData[i]));
        int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    
    const pcmBytes = new Uint8Array(int16.buffer);
    return {
        data: encode(pcmBytes),
        mimeType: `audio/pcm;rate=${targetSampleRate}`,
    };
};

interface VisualizerProps {
  mode: 'connecting' | 'active' | 'error';
  isSpeaking: boolean;
  micVolume: number;
}

const Visualizer: React.FC<VisualizerProps> = ({ mode, isSpeaking, micVolume }) => {
    const isActive = mode === 'active';
    const isError = mode === 'error';
    const isConnecting = mode === 'connecting';
    
    const micRingScale = isActive ? 1 + micVolume * 2.0 : 1;

    // Color Logic
    const mainColor = isError ? 'bg-red-500' : isActive ? 'bg-purple-500' : 'bg-yellow-500';
    const secondaryColor = isError ? 'bg-red-400' : isActive ? 'bg-cyan-500' : 'bg-amber-400';
    const ringBorder = isError ? 'border-red-500/20' : isActive ? 'border-cyan-500/10' : 'border-yellow-500/20';

    return (
        <div className="relative w-48 h-48 md:w-64 md:h-64 mx-auto flex items-center justify-center transition-all duration-500">
            {/* Spinning Ring */}
            <div
                className={`absolute w-[120%] h-[120%] rounded-full border border-dashed transition-all duration-1000 ${ringBorder} opacity-100 animate-[spin_10s_linear_infinite]`}
            ></div>

            {/* Model Ring (Outer) */}
            <div
                className={`absolute w-full h-full rounded-full ${mainColor} transition-all duration-500 ${
                    isConnecting ? 'animate-pulse opacity-50' : 
                    isActive ? 'opacity-20' : 'opacity-10'
                } ${isSpeaking ? 'scale-105' : 'scale-100'}`}
            ></div>

            {/* User Ring (Inner) */}
            <div
                className={`absolute w-3/4 h-3/4 rounded-full ${secondaryColor} transition-all duration-200 ${
                    isActive ? 'opacity-30' : 'opacity-0'
                }`}
                style={{ transform: `scale(${isSpeaking ? 0.8 : micRingScale})` }}
            ></div>

            {/* Core Identity */}
            <div className={`relative w-1/2 h-1/2 rounded-full shadow-2xl flex items-center justify-center transition-colors duration-500 ${
                isActive ? 'bg-gradient-to-br from-purple-600 to-cyan-500' :
                isError ? 'bg-red-800' :
                'bg-gradient-to-br from-yellow-600 to-amber-600'
            }`}>
                <div className="text-center">
                    <span className="text-white font-bold text-lg block tracking-widest drop-shadow-md">NORA</span>
                    <span className="text-[10px] uppercase tracking-wider text-white/80 font-medium mt-1">
                        {mode === 'connecting' ? 'INIT' : mode}
                    </span>
                </div>
            </div>
        </div>
    );
};

export const VoiceAssistant: React.FC = () => {
    const [mode, setMode] = useState<'connecting' | 'active' | 'error'>('connecting');
    const [isModelSpeaking, setIsModelSpeaking] = useState(false);
    const [statusMessage, setStatusMessage] = useState("Initializing...");
    const [micVolume, setMicVolume] = useState(0);
    const [isPermanentError, setIsPermanentError] = useState(false);

    // Refs for Audio & Session
    const sessionRef = useRef<Promise<any> | null>(null);
    const initializingRef = useRef(false);
    const mountedRef = useRef(true);
    const streamRef = useRef<MediaStream | null>(null);
    const audioContextRef = useRef<AudioContext | null>(null);
    const outputAudioContextRef = useRef<AudioContext | null>(null);
    const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);
    const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
    
    // Refs for Playback
    const nextStartTimeRef = useRef<number>(0);
    const playingSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
    
    // Reconnection Timer
    const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

    // --- Cleanup & Reset ---

    const stopAudioPlayback = useCallback(() => {
        if (outputAudioContextRef.current) {
            playingSourcesRef.current.forEach(source => {
                try { source.stop(); } catch (e) {}
            });
            playingSourcesRef.current.clear();
            nextStartTimeRef.current = 0;
            setIsModelSpeaking(false);
        }
    }, []);

    const stopSession = useCallback(async () => {
        // Clear timeout if pending
        if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
            reconnectTimeoutRef.current = null;
        }

        // Close Gemini Session
        if (sessionRef.current) {
            try {
                const session = await sessionRef.current;
                session.close();
            } catch (e) {
                console.debug("Session close error (ignored):", e);
            }
            sessionRef.current = null;
        }

        // Stop Media Stream Tracks
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
        }

        // Disconnect & Close Input Audio Context
        if (scriptProcessorRef.current) {
            try {
                scriptProcessorRef.current.disconnect();
                sourceRef.current?.disconnect();
            } catch (e) {}
            scriptProcessorRef.current = null;
            sourceRef.current = null;
        }
        if (audioContextRef.current) {
            try { await audioContextRef.current.close(); } catch (e) {}
            audioContextRef.current = null;
        }

        // Close Output Audio Context
        if (outputAudioContextRef.current) {
            stopAudioPlayback();
            try { await outputAudioContextRef.current.close(); } catch (e) {}
            outputAudioContextRef.current = null;
        }
    }, [stopAudioPlayback]);

    // Reconnect Logic
    const scheduleReconnect = useCallback((msg: string) => {
        if (!mountedRef.current) return;
        console.log("Scheduling reconnect:", msg);
        setStatusMessage(msg + " Retrying in 3s...");
        setMode('error');
        setMicVolume(0);
        
        // Clean up current session
        stopSession().then(() => {
            if (!mountedRef.current) return;
            // Wait 3s then try startGeminiSession again
            reconnectTimeoutRef.current = setTimeout(() => {
                startGeminiSession();
            }, 3000);
        });
    }, [stopSession]); 

    // --- Gemini Session Logic ---

    const startGeminiSession = async () => {
        // Prevent multiple calls
        if (sessionRef.current || initializingRef.current) return;
        initializingRef.current = true;

        if (mountedRef.current) {
            setMode('connecting');
            setStatusMessage("Connecting...");
            setIsPermanentError(false);
        }

        try {
            // 1. Check Browser Capability
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error("Browser API Not Supported");
            }

            // 2. Get Microphone Stream
            let stream: MediaStream;
            try {
                stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        channelCount: 1, // Force mono
                        echoCancellation: true,
                        autoGainControl: true,
                        noiseSuppression: true,
                    } 
                });
            } catch (err: any) {
                console.warn("Mic Access Error:", err);
                
                if (mountedRef.current) {
                    setMode('error');
                    setIsPermanentError(true);
                    
                    const msg = err.message || "";
                    if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError' || msg.includes("permission")) {
                        setStatusMessage("Microphone Permission Denied");
                    } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError' || msg.includes("not found")) {
                        setStatusMessage("No Microphone Found");
                    } else {
                        setStatusMessage("Microphone Error: " + (msg || "Unknown"));
                    }
                }
                return;
            }

            if (!mountedRef.current) {
                stream.getTracks().forEach(t => t.stop());
                return;
            }

            streamRef.current = stream;

            // 3. Initialize Audio Contexts
            let audioCtx: AudioContext;
            let outputCtx: AudioContext;
            try {
                const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
                
                // Input Context: Default settings (uses hardware rate)
                audioCtx = new AudioContextClass({ latencyHint: 'interactive' });
                
                // Output Context: Do NOT force sampleRate. 
                // Let the browser/device decide the native rate (e.g. 48000Hz).
                // We will decode the 24000Hz response audio into this context safely.
                outputCtx = new AudioContextClass({ latencyHint: 'interactive' });
                
            } catch (e) {
                console.error("Audio Context Creation Failed", e);
                stream.getTracks().forEach(track => track.stop());
                streamRef.current = null;
                
                if (mountedRef.current) {
                    setMode('error');
                    setIsPermanentError(true);
                    setStatusMessage("Audio Driver Error");
                }
                return;
            }
            
            audioContextRef.current = audioCtx;
            outputAudioContextRef.current = outputCtx;
            
            const actualSampleRate = audioCtx.sampleRate;
            console.log(`Audio Context initialized. Input Rate: ${actualSampleRate}, Output Rate: ${outputCtx.sampleRate}`);

            // 4. Connect to Gemini
            const ai = new GoogleGenAI({ apiKey: process.env.API_KEY! });
            
            const sessionPromise = ai.live.connect({
                model: 'gemini-2.5-flash-native-audio-preview-09-2025',
                config: {
                    responseModalities: [Modality.AUDIO],
                    speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Zephyr' } } },
                    systemInstruction: "You are Nora, a helpful and articulate AI assistant. Your name is Nora. If asked for your name, always answer 'Nora'. Do not mention Gemini or Google. Speak in Malayalam. You enjoy having long conversations and telling detailed stories. If asked to explain something or tell a story, provide a full, elaborate, and engaging response.",
                },
                callbacks: {
                    onopen: () => {
                        if (!mountedRef.current) return;
                        console.log("Gemini Connected");
                        setMode('active');
                        setStatusMessage("Listening...");
                        
                        // Setup Input Stream
                        try {
                            const source = audioCtx.createMediaStreamSource(stream);
                            sourceRef.current = source;
                            
                            // Buffer size 4096 is safe for smoother processing
                            const scriptProcessor = audioCtx.createScriptProcessor(4096, 1, 1);
                            scriptProcessorRef.current = scriptProcessor;

                            scriptProcessor.onaudioprocess = (e) => {
                                if (!mountedRef.current) return;
                                
                                // Process and Send Audio
                                try {
                                    const inputData = e.inputBuffer.getChannelData(0);
                                    
                                    // Prevent feedback: Silence output
                                    const outputData = e.outputBuffer.getChannelData(0);
                                    for (let i = 0; i < outputData.length; i++) {
                                        outputData[i] = 0;
                                    }

                                    // Calculate Volume for Visualizer
                                    let sum = 0;
                                    // Sample sparse data points for visualizer efficiency
                                    for (let i = 0; i < inputData.length; i += 16) { 
                                        sum += inputData[i] * inputData[i];
                                    }
                                    setMicVolume(Math.sqrt(sum / (inputData.length / 16)));

                                    // Create robust PCM blob (resamples if needed)
                                    const pcmBlob = createPcmBlobForGemini(inputData, actualSampleRate);
                                    
                                    sessionRef.current?.then(session => {
                                        session.sendRealtimeInput({ media: pcmBlob });
                                    }).catch(err => {
                                        // Silent fail - session might be reconnecting
                                    });
                                } catch (procError) {
                                    console.error("Audio Process Error", procError);
                                }
                            };

                            source.connect(scriptProcessor);
                            scriptProcessor.connect(audioCtx.destination);
                        } catch (setupError) {
                            console.error("Audio Pipeline Setup Error:", setupError);
                            scheduleReconnect("Audio Setup Failed");
                        }
                    },
                    onmessage: async (msg: LiveServerMessage) => {
                        if (!mountedRef.current) return;
                        // Handle Audio
                        const data = msg.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
                        if (data && outputCtx) {
                            setIsModelSpeaking(true);
                            try {
                                // Decode 24kHz audio from Gemini into the browser's native AudioContext
                                // The browser automatically handles resampling for playback.
                                const buffer = await decodeAudioData(decode(data), outputCtx, 24000, 1);
                                const source = outputCtx.createBufferSource();
                                source.buffer = buffer;
                                source.connect(outputCtx.destination);
                                source.onended = () => {
                                    if (playingSourcesRef.current) {
                                        playingSourcesRef.current.delete(source);
                                        if (playingSourcesRef.current.size === 0) setIsModelSpeaking(false);
                                    }
                                };
                                const now = outputCtx.currentTime;
                                const startTime = Math.max(nextStartTimeRef.current, now);
                                source.start(startTime);
                                nextStartTimeRef.current = startTime + buffer.duration;
                                playingSourcesRef.current.add(source);
                            } catch (e) {
                                console.error("Decoding error:", e);
                            }
                        }

                        if (msg.serverContent?.interrupted) {
                            stopAudioPlayback();
                        }
                    },
                    onclose: () => {
                        console.log("Session closed by server");
                        scheduleReconnect("Session ended.");
                    },
                    onerror: (err) => {
                        console.error("Session error:", err);
                        scheduleReconnect("Connection lost.");
                    }
                }
            });

            sessionPromise.catch(err => {
                console.error("Connection failed:", err);
                scheduleReconnect("Network Error.");
            });

            sessionRef.current = sessionPromise;

        } catch (err: any) {
            console.error("Start session failed (Outer):", err);
            if (mountedRef.current) {
                scheduleReconnect("Initialization Failed.");
            }
        } finally {
            initializingRef.current = false;
        }
    };

    // --- Lifecycle ---

    useEffect(() => {
        mountedRef.current = true;
        startGeminiSession();
        return () => {
            mountedRef.current = false;
            stopSession();
        };
    }, []);

    useEffect(() => {
        const handleInteraction = () => {
            if (audioContextRef.current?.state === 'suspended') {
                audioContextRef.current.resume();
            }
            if (outputAudioContextRef.current?.state === 'suspended') {
                outputAudioContextRef.current.resume();
            }
        };
        window.addEventListener('click', handleInteraction);
        window.addEventListener('keydown', handleInteraction);
        window.addEventListener('touchstart', handleInteraction);
        return () => {
            window.removeEventListener('click', handleInteraction);
            window.removeEventListener('keydown', handleInteraction);
            window.removeEventListener('touchstart', handleInteraction);
        };
    }, []);

    return (
        <div className="bg-gray-800/50 p-6 rounded-lg shadow-xl backdrop-blur-sm relative transition-all duration-500 border border-gray-700">
            <div className="text-center mb-6">
                 <Visualizer 
                    mode={mode}
                    isSpeaking={isModelSpeaking}
                    micVolume={micVolume}
                />
                 <div className="mt-8 min-h-[40px] flex flex-col items-center justify-center">
                    <p className={`font-semibold text-lg transition-colors duration-300 ${
                        mode === 'error' ? 'text-red-400' : 
                        mode === 'connecting' ? 'text-yellow-400' : 'text-cyan-400'
                    }`}>
                        {statusMessage}
                    </p>
                    {isPermanentError && (
                        <button 
                            onClick={() => {
                                setIsPermanentError(false);
                                startGeminiSession();
                            }}
                            className="mt-2 px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded text-sm text-white transition-colors border border-gray-600"
                        >
                            Retry Connection
                        </button>
                    )}
                </div>
            </div>
            
            <div className="text-center text-gray-500 text-xs mt-4 border-t border-gray-700/50 pt-4">
                 {mode === 'connecting' ? (
                     <p>Establishing Connection...</p>
                 ) : mode === 'error' ? (
                     <p>Checking System...</p>
                 ) : (
                     <p>Nora is listening • Malayalam</p>
                 )}
            </div>
        </div>
    );
};
