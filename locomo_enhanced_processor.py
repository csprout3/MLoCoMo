"""
Enhanced MLOCOMO Processor aligned with LOCOMO paper methodology
https://arxiv.org/abs/2402.17753

This processor synthesizes multi-session conversations from single videos
to create LOCOMO-compliant datasets with:
- Multiple speakers (synthesized personas)
- Multiple sessions (temporal progression)
- Event graphs and persona grounding
- 300+ turns across 35+ sessions
"""

import json
import asyncio
import os
import subprocess
import logging
import time
import base64
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import uuid
from datetime import datetime, timedelta
import random

import cv2
import numpy as np
import whisper
import openai
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

@dataclass
class Persona:
    """Persona for LOCOMO-style multi-speaker conversations"""
    name: str
    age: int
    background: str
    interests: List[str]
    communication_style: str
    relationship_to_topic: str
    personality_traits: List[str]
    speech_characteristics: str

@dataclass
class EventNode:
    """Event node for temporal event graph"""
    event_id: str
    timestamp: float
    description: str
    participants: List[str]
    event_type: str
    emotional_tone: str
    key_objects: List[str]

@dataclass
class ConversationSession:
    """Single conversation session following LOCOMO structure"""
    session_id: str
    session_number: int
    timestamp: str
    participants: List[str]
    turns: List[Dict]
    session_topic: str
    emotional_arc: str
    key_events: List[str]

@dataclass
class LOCOMOConversation:
    """Complete LOCOMO conversation with multiple sessions"""
    conversation_id: str
    personas: Dict[str, Persona]
    event_graph: List[EventNode]
    sessions: List[ConversationSession]
    qa_pairs: List[Dict]
    metadata: Dict

class LOCOMOEnhancedProcessor:
    """Enhanced processor that creates LOCOMO-compliant multi-session conversations"""
    
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.setup_complete = False
        
        # LOCOMO requirements
        self.target_sessions = 35  # LOCOMO uses 35 sessions
        self.target_turns_per_session = 8-12  # ~300 total turns
        self.recall_types = [
            "single_hop", "multi_hop", "temporal", 
            "commonsense", "adversarial", "cross_modal"
        ]
        
    def setup_models(self):
        """Load required models"""
        print("🔄 Loading models...")
        
        # Load Whisper for transcription
        self.whisper_model = whisper.load_model("base")
        
        # Load sentence transformer for embeddings
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.setup_complete = True
        print("✅ Models loaded successfully!")
    
    def create_personas_from_video(self, video_content: str) -> Dict[str, Persona]:
        """Create multiple personas based on video content and audience reactions"""
        
        persona_prompt = f"""
        Based on this TED talk content: "{video_content[:2000]}..."
        
        Create 3-4 distinct personas who would be in the audience and later discuss this talk.
        Each persona should have:
        1. Different backgrounds and perspectives on the topic
        2. Different communication styles and personalities
        3. Different relationships to vulnerability/leadership themes
        4. Distinct speech patterns and characteristics
        
        Format as JSON:
        {{
            "persona_1": {{
                "name": "Name",
                "age": 25-45,
                "background": "Professional background",
                "interests": ["interest1", "interest2"],
                "communication_style": "formal/casual/analytical",
                "relationship_to_topic": "personal/professional/academic",
                "personality_traits": ["trait1", "trait2"],
                "speech_characteristics": "speaks quickly/slowly, uses metaphors, etc."
            }},
            ...
        }}
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": persona_prompt}],
            temperature=0.8
        )
        
        try:
            personas_data = json.loads(response.choices[0].message.content)
            personas = {}
            for key, data in personas_data.items():
                personas[key] = Persona(**data)
            return personas
        except:
            # Fallback personas
            return {
                "alex": Persona(
                    name="Alex", age=28, background="Software Engineer",
                    interests=["technology", "personal growth"],
                    communication_style="analytical", relationship_to_topic="professional",
                    personality_traits=["curious", "logical"], speech_characteristics="speaks clearly, uses technical terms"
                ),
                "maya": Persona(
                    name="Maya", age=35, background="Therapist",
                    interests=["psychology", "human behavior"],
                    communication_style="empathetic", relationship_to_topic="professional",
                    personality_traits=["compassionate", "intuitive"], speech_characteristics="speaks softly, uses metaphors"
                ),
                "jordan": Persona(
                    name="Jordan", age=42, background="Manager",
                    interests=["leadership", "team building"],
                    communication_style="direct", relationship_to_topic="professional",
                    personality_traits=["confident", "practical"], speech_characteristics="speaks assertively, uses business terms"
                )
            }
    
    def create_event_graph(self, video_segments: List[Dict], personas: Dict[str, Persona]) -> List[EventNode]:
        """Create temporal event graph from video content"""
        
        events = []
        for i, segment in enumerate(video_segments):
            # Create main event from video segment
            main_event = EventNode(
                event_id=f"event_{i:03d}",
                timestamp=segment.get('start_time', i * 30),
                description=segment.get('transcript', '')[:200],
                participants=["speaker"],  # Brené Brown
                event_type="presentation",
                emotional_tone="inspiring",
                key_objects=segment.get('key_objects', [])
            )
            events.append(main_event)
            
            # Create discussion events between personas
            if i % 3 == 0:  # Every 3rd segment
                discussion_event = EventNode(
                    event_id=f"discussion_{i:03d}",
                    timestamp=segment.get('start_time', i * 30) + 5,
                    description=f"Discussion about: {segment.get('transcript', '')[:100]}...",
                    participants=list(personas.keys()),
                    event_type="discussion",
                    emotional_tone="thoughtful",
                    key_objects=["conversation", "reflection"]
                )
                events.append(discussion_event)
        
        return events
    
    def generate_session_conversation(self, session_num: int, personas: Dict[str, Persona], 
                                    event_graph: List[EventNode], previous_sessions: List[ConversationSession]) -> ConversationSession:
        """Generate a single conversation session following LOCOMO methodology"""
        
        # Select relevant events for this session
        session_events = event_graph[session_num*2:(session_num+1)*2]
        
        # Build context from previous sessions
        previous_context = ""
        if previous_sessions:
            recent_topics = [s.session_topic for s in previous_sessions[-3:]]
            previous_context = f"Previous discussions covered: {', '.join(recent_topics)}"
        
        # Generate conversation turns
        conversation_prompt = f"""
        Generate a natural conversation between these personas about the TED talk content.
        
        Personas: {[f"{p.name} ({p.background})" for p in personas.values()]}
        
        Session {session_num + 1} context: {previous_context}
        
        Current events to discuss:
        {[f"- {e.description[:100]}..." for e in session_events]}
        
        Guidelines:
        1. Generate 8-12 conversation turns
        2. Each persona should speak 2-3 times
        3. Include references to specific moments from the talk
        4. Show different perspectives and reactions
        5. Include personal anecdotes and connections
        6. Maintain character consistency
        7. Reference previous discussions naturally
        
        Format as JSON:
        [
            {{
                "speaker": "persona_name",
                "text": "conversation text",
                "timestamp": "session_timestamp",
                "references_video": true/false,
                "emotional_tone": "excited/thoughtful/concerned/etc"
            }},
            ...
        ]
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": conversation_prompt}],
            temperature=0.8
        )
        
        try:
            turns_data = json.loads(response.choices[0].message.content)
        except:
            # Fallback conversation
            turns_data = [
                {"speaker": list(personas.keys())[0], "text": f"Session {session_num + 1} discussion about the talk", "timestamp": f"session_{session_num + 1}", "references_video": True, "emotional_tone": "thoughtful"}
            ]
        
        # Create session
        session = ConversationSession(
            session_id=f"session_{session_num:03d}",
            session_number=session_num + 1,
            timestamp=datetime.now().isoformat(),
            participants=list(personas.keys()),
            turns=turns_data,
            session_topic=f"Discussion about vulnerability and leadership",
            emotional_arc="progressive engagement",
            key_events=[e.event_id for e in session_events]
        )
        
        return session
    
    def generate_qa_pairs(self, conversation: LOCOMOConversation) -> List[Dict]:
        """Generate QA pairs following LOCOMO methodology"""
        
        qa_pairs = []
        
        # Flatten all turns for context
        all_turns = []
        for session in conversation.sessions:
            all_turns.extend(session.turns)
        
        # Generate questions for each recall type
        for recall_type in self.recall_types:
            qa_prompt = f"""
            Based on this multi-session conversation about a TED talk:
            
            Sessions: {len(conversation.sessions)}
            Total turns: {len(all_turns)}
            Participants: {list(conversation.personas.keys())}
            
            Generate 5 {recall_type} questions that test:
            - Single-hop: Direct recall of specific information
            - Multi-hop: Reasoning across multiple conversation elements
            - Temporal: Time-based understanding and ordering
            - Commonsense: Contextual understanding and practical reasoning
            - Adversarial: Robustness under challenging conditions
            - Cross-modal: Integration of different information types
            
            Format as JSON:
            [
                {{
                    "question": "question text",
                    "answer": "ground truth answer",
                    "recall_type": "{recall_type}",
                    "difficulty": "easy/medium/hard",
                    "requires_cross_session": true/false,
                    "evidence_turns": ["session_id:turn_index"],
                    "reasoning_steps": ["step1", "step2"]
                }},
                ...
            ]
            """
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": qa_prompt}],
                    temperature=0.7
                )
                
                qa_data = json.loads(response.choices[0].message.content)
                if isinstance(qa_data, list):
                    qa_pairs.extend(qa_data)
                else:
                    qa_pairs.append(qa_data)
                    
            except Exception as e:
                print(f"⚠️  Error generating {recall_type} questions: {e}")
                continue
        
        return qa_pairs
    
    def process_video_to_locomo(self, video_path: str) -> LOCOMOConversation:
        """Process video into LOCOMO-compliant conversation"""
        
        print("🔄 Processing video to LOCOMO format...")
        
        # Transcribe video
        print("📝 Transcribing video...")
        result = self.whisper_model.transcribe(video_path)
        
        # Create personas
        print("👥 Creating personas...")
        personas = self.create_personas_from_video(result["text"])
        
        # Create event graph
        print("📊 Creating event graph...")
        # Simulate video segments for event creation
        video_segments = [{"transcript": result["text"][i:i+500], "start_time": i*30, "key_objects": ["speaker", "audience"]} for i in range(0, len(result["text"]), 500)]
        event_graph = self.create_event_graph(video_segments, personas)
        
        # Generate multiple sessions
        print(f"💬 Generating {self.target_sessions} conversation sessions...")
        sessions = []
        for session_num in range(self.target_sessions):
            if session_num % 5 == 0:
                print(f"   Session {session_num + 1}/{self.target_sessions}")
            
            session = self.generate_session_conversation(session_num, personas, event_graph, sessions)
            sessions.append(session)
        
        # Generate QA pairs
        print("❓ Generating QA pairs...")
        conversation = LOCOMOConversation(
            conversation_id=str(uuid.uuid4())[:8],
            personas=personas,
            event_graph=event_graph,
            sessions=sessions,
            qa_pairs=[],
            metadata={"source_video": video_path, "processing_timestamp": datetime.now().isoformat()}
        )
        
        qa_pairs = self.generate_qa_pairs(conversation)
        conversation.qa_pairs = qa_pairs
        
        print(f"✅ Generated {len(qa_pairs)} QA pairs!")
        
        return conversation

# Example usage
def main():
    """Example usage of LOCOMO Enhanced Processor"""
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OpenAI API key required")
        return
    
    # Initialize processor
    processor = LOCOMOEnhancedProcessor(api_key)
    processor.setup_models()
    
    # Process video (replace with actual video path)
    video_path = "sample_video.mp4"  # Replace with your video
    conversation = processor.process_video_to_locomo(video_path)
    
    # Save results
    output_file = "locomo_enhanced_dataset.json"
    with open(output_file, 'w') as f:
        json.dump(asdict(conversation), f, indent=2, default=str)
    
    print(f"✅ LOCOMO dataset saved to {output_file}")
    print(f"📊 Statistics:")
    print(f"   Sessions: {len(conversation.sessions)}")
    print(f"   Total turns: {sum(len(s.turns) for s in conversation.sessions)}")
    print(f"   QA pairs: {len(conversation.qa_pairs)}")
    print(f"   Personas: {len(conversation.personas)}")

if __name__ == "__main__":
    main()
