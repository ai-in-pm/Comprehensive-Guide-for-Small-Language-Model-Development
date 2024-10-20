import React from 'react';
import { Server } from 'lucide-react';

const DeploymentAndScaling: React.FC<{ isActive: boolean }> = ({ isActive }) => {
  if (!isActive) return null;

  return (
    <div className="space-y-6">
      <h2 className="text-3xl font-bold text-gray-800 flex items-center">
        <Server className="mr-2" />
        Deployment and Scaling
      </h2>
      <p className="text-gray-600">
        Deploying and scaling a Small Language Model (SLM) requires careful consideration of infrastructure, performance, and resource management. This section covers key aspects of deploying and scaling your SLM effectively, with a focus on efficiency and reliability.
      </p>
      
      {/* ... (previous content remains the same) ... */}
      
      <div className="mt-8 p-6 bg-blue-50 rounded-lg border border-blue-200">
        <h4 className="text-xl font-semibold text-blue-800 mb-4">Code Snippet: Docker Compose for SLM Deployment</h4>
        <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
          <code>{`
version: '3.8'

services:
  slm-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models/slm_model
    volumes:
      - ./models:/app/models
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 10s
      restart_policy:
        condition: on-failure
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:latest
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - slm-api

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"

volumes:
  models:
          `}</code>
        </pre>
      </div>
      
      <div className="mt-8 p-6 bg-green-50 rounded-lg border border-green-200">
        <h4 className="text-xl font-semibold text-green-800 mb-4">Best Practices for SLM Deployment and Scaling</h4>
        <ul className="list-disc list-inside text-green-700 space-y-2">
          <li>Use auto-scaling based on traffic patterns and resource utilization</li>
          <li>Implement robust monitoring and logging for performance tracking</li>
          <li>Use Content Delivery Networks (CDNs) for global distribution</li>
          <li>Implement blue-green or canary deployment strategies for updates</li>
          <li>Use serverless architectures for cost-effective scaling of certain components</li>
          <li>Implement circuit breakers to handle failures gracefully</li>
          <li>Use database sharding for improved data management at scale</li>
          <li>Implement rate limiting to prevent abuse and ensure fair usage</li>
          <li>Use message queues for asynchronous processing of long-running tasks</li>
          <li>Implement proper error handling and retry mechanisms for improved resilience</li>
        </ul>
      </div>
      
      <div className="mt-8 p-6 bg-yellow-50 rounded-lg border border-yellow-200">
        <h4 className="text-xl font-semibold text-yellow-800 mb-4">Challenges and Strategies for SLM Deployment and Scaling</h4>
        <ul className="list-disc list-inside text-yellow-700 space-y-2">
          <li>Cold start latency: Implement model caching and warm-up strategies</li>
          <li>Resource constraints: Use efficient model compression techniques like quantization and pruning</li>
          <li>High concurrency: Implement efficient batching and request queuing mechanisms</li>
          <li>Data privacy concerns: Use federated learning or on-device inference where appropriate</li>
          <li>Model updates: Implement seamless model versioning and update strategies</li>
          <li>Cost management: Optimize resource allocation and implement usage-based pricing</li>
          <li>Latency requirements: Use edge computing for low-latency applications</li>
          <li>Scalability across regions: Implement multi-region deployment with data replication</li>
          <li>Monitoring and debugging: Use distributed tracing and advanced logging techniques</li>
          <li>Compliance and regulations: Implement data governance and auditing mechanisms</li>
        </ul>
      </div>
    </div>
  );
};

export default DeploymentAndScaling;