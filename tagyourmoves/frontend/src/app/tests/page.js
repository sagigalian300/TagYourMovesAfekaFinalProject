"use client";

import { useState } from 'react';
import Link from 'next/link';
import TestCard from '@/components/tests/TestCard';
import ObjectDetection from '@/components/tests/ObjectDetection';
import TeamAssignment from '@/components/tests/TeamAssignment';

export default function TestsPage() {
  const [activeTab, setActiveTab] = useState('object-detection');

  return (
    <main className="min-h-screen bg-gray-50 p-6 md:p-12 font-sans">
      {/* הקטנתי מעט את הרוחב המקסימלי כדי שכרטיס בודד לא יראה מרוח מדי */}
      <div className="max-w-4xl mx-auto">
        
        {/* Header Section */}
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8 pb-6 border-b border-gray-200">
          <div>
            <h1 className="text-3xl font-extrabold text-gray-900 tracking-tight">
              System Diagnostics
            </h1>
            <p className="text-sm text-gray-500 mt-1 font-medium">
              Tag Your Moves - Integration Testing Environment
            </p>
          </div>
          
          <Link href="/" className="mt-4 md:mt-0 px-4 py-2 text-sm font-medium text-gray-600 bg-white border border-gray-300 rounded-md hover:bg-gray-50 transition-colors">
            Exit Dashboard
          </Link>
        </div>

        {/* Tabs Navigation */}
        <div className="flex space-x-6 border-b border-gray-200 mb-8">
          <button
            onClick={() => setActiveTab('object-detection')}
            className={`pb-3 text-sm font-medium transition-colors relative ${
              activeTab === 'object-detection'
                ? 'text-black'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            Object Detection
            {activeTab === 'object-detection' && (
              <span className="absolute bottom-0 left-0 w-full h-0.5 bg-black rounded-t-md" />
            )}
          </button>
          
          <button
            onClick={() => setActiveTab('team-assignment')}
            className={`pb-3 text-sm font-medium transition-colors relative ${
              activeTab === 'team-assignment'
                ? 'text-black'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            Team Assignment
            {activeTab === 'team-assignment' && (
              <span className="absolute bottom-0 left-0 w-full h-0.5 bg-black rounded-t-md" />
            )}
          </button>
        </div>

        {/* Active Test Content */}
        <div className="transition-opacity duration-300">
          {activeTab === 'object-detection' && (
            <TestCard 
              title="Object Detection MS"
              description="Test the Object Detection MS"
              icon={<svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"></path></svg>}
            >
              <ObjectDetection />
            </TestCard>
          )}

          {activeTab === 'team-assignment' && (
            <TestCard 
              title="Team Assignment MS"
              description="Test the Team Assignment MS"
              icon={<svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"></path></svg>}
            >
              <TeamAssignment />
            </TestCard>
          )}
        </div>

      </div>
    </main>
  );
}