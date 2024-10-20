import React from 'react';
import { Zap } from 'lucide-react';

const UserInterface: React.FC<{ isActive: boolean }> = ({ isActive }) => {
  if (!isActive) return null;

  return (
    <div className="space-y-6">
      <h2 className="text-3xl font-bold text-gray-800 flex items-center">
        <Zap className="mr-2" />
        User Interface
      </h2>
      <p className="text-gray-600">
        Developing an intuitive and efficient user interface is crucial for showcasing and interacting with your Small Language Model (SLM). This section covers key aspects of creating a user-friendly interface that highlights the capabilities of your SLM while ensuring a smooth user experience.
      </p>
      
      <h3 className="text-2xl font-semibold text-gray-700">1. Web-based Interface</h3>
      <p className="text-gray-600">
        Develop a responsive web application for interacting with your SLM:
      </p>
      <ul className="list-disc list-inside text-gray-600 ml-4">
        <li>Use modern frontend frameworks like React, Vue, or Svelte for building interactive UIs</li>
        <li>Implement responsive design for compatibility across devices (desktop, tablet, mobile)</li>
        <li>Utilize CSS frameworks like Tailwind CSS for rapid UI development</li>
        <li>Implement progressive enhancement for better accessibility and performance</li>
      </ul>
      
      <h3 className="text-2xl font-semibold text-gray-700">2. Simulation Page</h3>
      <p className="text-gray-600">
        Create an interactive simulation page to demonstrate the capabilities of your SLM in real-time:
      </p>
      <ul className="list-disc list-inside text-gray-600 ml-4">
        <li>Implement a playground area for users to input custom text and see model outputs</li>
        <li>Provide pre-set examples to showcase different capabilities of the SLM</li>
        <li>Include options to adjust model parameters (e.g., temperature, top-k sampling)</li>
        <li>Implement real-time updates to showcase the model's processing speed</li>
      </ul>
      
      <h3 className="text-2xl font-semibold text-gray-700">3. Performance Visualizations</h3>
      <p className="text-gray-600">
        Implement visualizations to showcase model performance and outputs:
      </p>
      <ul className="list-disc list-inside text-gray-600 ml-4">
        <li>Use charts and graphs to display metrics like accuracy, latency, and resource usage</li>
        <li>Implement word clouds or highlight mechanisms to show important words/phrases in outputs</li>
        <li>Create comparative visualizations to show SLM performance against baselines or larger models</li>
        <li>Use animation to illustrate the model's decision-making process</li>
      </ul>
      
      <h3 className="text-2xl font-semibold text-gray-700">4. Chat Interface</h3>
      <p className="text-gray-600">
        Design an intuitive chat interface for conversational interactions with your SLM:
      </p>
      <ul className="list-disc list-inside text-gray-600 ml-4">
        <li>Implement a message thread view with clear distinction between user and SLM messages</li>
        <li>Add typing indicators to show when the SLM is processing a response</li>
        <li>Include options for users to rate or provide feedback on SLM responses</li>
        <li>Implement message persistence for continuing conversations across sessions</li>
      </ul>
      
      <h3 className="text-2xl font-semibold text-gray-700">5. Accessibility Features</h3>
      <p className="text-gray-600">
        Implement accessibility features to ensure the interface is usable by diverse user groups:
      </p>
      <ul className="list-disc list-inside text-gray-600 ml-4">
        <li>Ensure proper contrast ratios and font sizes for readability</li>
        <li>Implement keyboard navigation for all interactive elements</li>
        <li>Add ARIA labels and roles for screen reader compatibility</li>
        <li>Provide text alternatives for non-text content</li>
      </ul>
      
      <h3 className="text-2xl font-semibold text-gray-700">6. Mobile Responsiveness</h3>
      <p className="text-gray-600">
        Develop mobile-responsive designs for cross-platform compatibility:
      </p>
      <ul className="list-disc list-inside text-gray-600 ml-4">
        <li>Use flexible layouts and CSS Grid/Flexbox for adaptive designs</li>
        <li>Implement touch-friendly interfaces for mobile devices</li>
        <li>Optimize asset loading for faster mobile performance</li>
        <li>Consider developing a Progressive Web App (PWA) for enhanced mobile experience</li>
      </ul>
      
      <h3 className="text-2xl font-semibold text-gray-700">7. Real-time Feedback</h3>
      <p className="text-gray-600">
        Implement real-time feedback mechanisms to enhance user interactions:
      </p>
      <ul className="list-disc list-inside text-gray-600 ml-4">
        <li>Use WebSockets or Server-Sent Events for real-time updates</li>
        <li>Implement progress indicators for long-running tasks</li>
        <li>Provide immediate visual feedback for user actions</li>
        <li>Use toast notifications for important updates or alerts</li>
      </ul>
      
      <div className="mt-8 p-6 bg-blue-50 rounded-lg border border-blue-200">
        <h4 className="text-xl font-semibold text-blue-800 mb-4">Code Snippet: React Component for SLM Chat Interface</h4>
        <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
          <code>{`
import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    setIsLoading(true);
    setMessages(prev => [...prev, { text: input, sender: 'user' }]);
    setInput('');

    try {
      const response = await axios.post('/api/chat', { message: input });
      setMessages(prev => [...prev, { text: response.data.reply, sender: 'bot' }]);
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, { text: 'Sorry, an error occurred.', sender: 'bot' }]);
    }

    setIsLoading(false);
  };

  return (
    <div className="max-w-md mx-auto mt-10 p-6 bg-white rounded-lg shadow-xl">
      <div className="h-96 overflow-y-auto mb-4 p-4 bg-gray-100 rounded">
        {messages.map((msg, index) => (
          <div key={index} className={\`mb-2 \${msg.sender === 'user' ? 'text-right' : 'text-left'}\`}>
            <span className={\`inline-block p-2 rounded \${
              msg.sender === 'user' ? 'bg-blue-500 text-white' : 'bg-gray-300'
            }\`}>
              {msg.text}
            </span>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      <form onSubmit={handleSubmit} className="flex">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          className="flex-grow p-2 border rounded-l focus:outline-none focus:ring-2 focus:ring-blue-500"
          placeholder="Type your message..."
          disabled={isLoading}
        />
        <button
          type="submit"
          className="bg-blue-500 text-white p-2 rounded-r hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
          disabled={isLoading}
        >
          {isLoading ? 'Sending...' : 'Send'}
        </button>
      </form>
    </div>
  );
};

export default ChatInterface;
          `}</code>
        </pre>
      </div>
      
      <div className="mt-8 p-6 bg-green-50 rounded-lg border border-green-200">
        <h4 className="text-xl font-semibold text-green-800 mb-4">Best Practices for SLM User Interface</h4>
        <ul className="list-disc list-inside text-green-700 space-y-2">
          <li>Use progressive loading techniques to improve perceived performance</li>
          <li>Implement error handling and user-friendly error messages</li>
          <li>Use WebSockets for real-time communication when appropriate</li>
          <li>Implement proper state management (e.g., using Redux or Context API)</li>
          <li>Use lazy loading for components to improve initial load time</li>
          <li>Implement proper keyboard navigation for accessibility</li>
          <li>Use ARIA attributes to enhance accessibility for screen readers</li>
          <li>Implement dark mode and other user preference options</li>
          <li>Use skeleton screens or loading indicators for better user experience</li>
          <li>Implement proper form validation and error handling</li>
        </ul>
      </div>
      
      <div className="mt-8 p-6 bg-yellow-50 rounded-lg border border-yellow-200">
        <h4 className="text-xl font-semibold text-yellow-800 mb-4">Challenges and Strategies for SLM User Interface</h4>
        <ul className="list-disc list-inside text-yellow-700 space-y-2">
          <li>Slow response times: Implement optimistic UI updates and caching strategies</li>
          <li>Complex interactions: Use progressive disclosure techniques and guided tours</li>
          <li>Cross-browser compatibility: Implement thorough testing and use polyfills when necessary</li>
          <li>Accessibility compliance: Conduct regular audits and user testing with assistive technologies</li>
          <li>Performance on low-end devices: Optimize bundle size and use code splitting techniques</li>
          <li>Handling large datasets: Implement virtualization for long lists and pagination for large result sets</li>
          <li>Internationalization: Use i18n libraries and design with text expansion in mind</li>
          <li>Consistent design: Implement a design system and use component libraries</li>
          <li>User onboarding: Create interactive tutorials and contextual help features</li>
          <li>Privacy concerns: Implement clear data usage policies and user consent mechanisms</li>
        </ul>
      </div>
    </div>
  );
};

export default UserInterface;