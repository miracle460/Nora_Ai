import React from 'react';
import { VoiceAssistant } from './components/VoiceAssistant';

const App: React.FC = () => {
  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 font-sans">
      <div className="container mx-auto p-4 max-w-4xl">
        <header className="text-center mb-8">
          <h1 className="text-4xl md:text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-cyan-400">
            Nora AI Assistant
          </h1>
        </header>

        <main>
          <VoiceAssistant />
        </main>
      </div>
    </div>
  );
};

export default App;