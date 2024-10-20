import React, { useState } from 'react';
import { Book, Code, Database, Cpu, Sliders, Layers, Zap, Server, RefreshCw, Shield, Microscope } from 'lucide-react';
import ModelConceptualization from './components/ModelConceptualization';
import DataPreparation from './components/DataPreparation';
import ModelImplementation from './components/ModelImplementation';
import TrainingProcess from './components/TrainingProcess';
import EvaluationAndFineTuning from './components/EvaluationAndFineTuning';
import MultiModalCapabilities from './components/MultiModalCapabilities';
import ApiDevelopment from './components/ApiDevelopment';
import UserInterface from './components/UserInterface';
import TuringTestChallenge from './components/TuringTestChallenge';
import DeploymentAndScaling from './components/DeploymentAndScaling';
import ContinuousImprovement from './components/ContinuousImprovement';
import EthicalConsiderations from './components/EthicalConsiderations';
import PerformanceOptimization from './components/PerformanceOptimization';
import RobustnessAndSecurity from './components/RobustnessAndSecurity';
import AdvancedCapabilities from './components/AdvancedCapabilities';
import DownloadButton from './components/DownloadButton';

const sections = [
  { id: 'conceptualization', title: 'Model Conceptualization', icon: Book, component: ModelConceptualization },
  { id: 'data-preparation', title: 'Data Preparation', icon: Database, component: DataPreparation },
  { id: 'implementation', title: 'Model Implementation', icon: Code, component: ModelImplementation },
  { id: 'training', title: 'Training Process', icon: Cpu, component: TrainingProcess },
  { id: 'evaluation', title: 'Evaluation and Fine-tuning', icon: Sliders, component: EvaluationAndFineTuning },
  { id: 'multi-modal', title: 'Multi-modal Capabilities', icon: Layers, component: MultiModalCapabilities },
  { id: 'api', title: 'API Development', icon: Server, component: ApiDevelopment },
  { id: 'ui', title: 'User Interface', icon: Zap, component: UserInterface },
  { id: 'turing-test', title: 'Turing Test Challenge', icon: Microscope, component: TuringTestChallenge },
  { id: 'deployment', title: 'Deployment and Scaling', icon: Server, component: DeploymentAndScaling },
  { id: 'improvement', title: 'Continuous Improvement', icon: RefreshCw, component: ContinuousImprovement },
  { id: 'ethics', title: 'Ethical Considerations', icon: Shield, component: EthicalConsiderations },
  { id: 'optimization', title: 'Performance Optimization', icon: Zap, component: PerformanceOptimization },
  { id: 'security', title: 'Robustness and Security', icon: Shield, component: RobustnessAndSecurity },
  { id: 'advanced', title: 'Advanced Capabilities', icon: Cpu, component: AdvancedCapabilities },
];

function App() {
  const [activeSection, setActiveSection] = useState(sections[0].id);

  return (
    <div className="min-h-screen bg-gray-100 flex">
      <nav className="w-64 bg-white shadow-lg">
        <div className="p-4">
          <h1 className="text-2xl font-bold text-gray-800">SLM Development Guide</h1>
        </div>
        <ul className="space-y-2 p-4">
          {sections.map((section) => (
            <li key={section.id}>
              <button
                onClick={() => setActiveSection(section.id)}
                className={`flex items-center space-x-2 w-full p-2 rounded ${
                  activeSection === section.id ? 'bg-blue-500 text-white' : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                <section.icon size={20} />
                <span>{section.title}</span>
              </button>
            </li>
          ))}
        </ul>
      </nav>
      <main className="flex-1 p-8 overflow-auto">
        {sections.map((section) => (
          <section.component key={section.id} isActive={activeSection === section.id} />
        ))}
      </main>
      <DownloadButton />
    </div>
  );
}

export default App;