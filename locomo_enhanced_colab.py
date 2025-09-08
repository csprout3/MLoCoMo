"""
LOCOMO Enhanced Processing for Google Colab
Based on: https://arxiv.org/abs/2402.17753

This creates LOCOMO-compliant multi-session conversations from single videos
"""

# Add this cell to your Colab notebook after the existing setup

def create_locomo_enhanced_processor():
    """Create enhanced processor that follows LOCOMO methodology"""
    
    class LOCOMOEnhancedProcessor:
        def __init__(self, api_key: str):
            self.client = openai.OpenAI(api_key=api_key)
            self.setup_complete = False
            
            # LOCOMO requirements from paper
            self.target_sessions = 35  # LOCOMO uses 35 sessions
            self.target_turns_per_session = 8-12  # ~300 total turns
            self.recall_types = [
                "single_hop", "multi_hop", "temporal", 
                "commonsense", "adversarial", "cross_modal"
            ]
        
        def setup_models(self):
            """Load required models"""
            print("🔄 Loading models for LOCOMO processing...")
            
            # Load Whisper for transcription
            self.whisper_model = whisper.load_model("base")
            
            # Load sentence transformer for embeddings
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            self.setup_complete = True
            print("✅ Models loaded successfully!")
        
        def create_personas_from_video(self, video_content: str):
            """Create multiple personas based on video content"""
            
            persona_prompt = f"""
            Based on this TED talk content: "{video_content[:2000]}..."
            
            Create 3-4 distinct personas who would be in the audience and later discuss this talk.
            Each persona should have different backgrounds, perspectives, and communication styles.
            
            Format as JSON with personas having: name, age, background, interests, 
            communication_style, relationship_to_topic, personality_traits, speech_characteristics
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": persona_prompt}],
                temperature=0.8
            )
            
            try:
                return json.loads(response.choices[0].message.content)
            except:
                # Fallback personas
                return {
                    "alex": {"name": "Alex", "age": 28, "background": "Software Engineer", "interests": ["technology", "personal growth"], "communication_style": "analytical", "relationship_to_topic": "professional", "personality_traits": ["curious", "logical"], "speech_characteristics": "speaks clearly, uses technical terms"},
                    "maya": {"name": "Maya", "age": 35, "background": "Therapist", "interests": ["psychology", "human behavior"], "communication_style": "empathetic", "relationship_to_topic": "professional", "personality_traits": ["compassionate", "intuitive"], "speech_characteristics": "speaks softly, uses metaphors"},
                    "jordan": {"name": "Jordan", "age": 42, "background": "Manager", "interests": ["leadership", "team building"], "communication_style": "direct", "relationship_to_topic": "professional", "personality_traits": ["confident", "practical"], "speech_characteristics": "speaks assertively, uses business terms"}
                }
        
        def generate_session_conversation(self, session_num: int, personas: dict, video_content: str, previous_sessions: list):
            """Generate a single conversation session"""
            
            # Build context from previous sessions
            previous_context = ""
            if previous_sessions:
                recent_topics = [s.get('session_topic', '') for s in previous_sessions[-3:]]
                previous_context = f"Previous discussions covered: {', '.join(recent_topics)}"
            
            # Generate conversation turns
            conversation_prompt = f"""
            Generate a natural conversation between these personas about the TED talk content.
            
            Personas: {[f"{p['name']} ({p['background']})" for p in personas.values()]}
            
            Session {session_num + 1} context: {previous_context}
            
            Video content to discuss: {video_content[:1000]}...
            
            Guidelines:
            1. Generate 8-12 conversation turns
            2. Each persona should speak 2-3 times
            3. Include references to specific moments from the talk
            4. Show different perspectives and reactions
            5. Include personal anecdotes and connections
            6. Maintain character consistency
            
            Format as JSON array of turns with speaker, text, timestamp, references_video, emotional_tone
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
            
            return {
                "session_id": f"session_{session_num:03d}",
                "session_number": session_num + 1,
                "timestamp": datetime.now().isoformat(),
                "participants": list(personas.keys()),
                "turns": turns_data,
                "session_topic": f"Discussion about vulnerability and leadership",
                "emotional_arc": "progressive engagement"
            }
        
        def generate_qa_pairs(self, sessions: list, personas: dict):
            """Generate QA pairs following LOCOMO methodology"""
            
            qa_pairs = []
            
            # Flatten all turns for context
            all_turns = []
            for session in sessions:
                all_turns.extend(session.get('turns', []))
            
            # Generate questions for each recall type
            for recall_type in self.recall_types:
                qa_prompt = f"""
                Based on this multi-session conversation about a TED talk:
                
                Sessions: {len(sessions)}
                Total turns: {len(all_turns)}
                Participants: {list(personas.keys())}
                
                Generate 5 {recall_type} questions that test:
                - Single-hop: Direct recall of specific information
                - Multi-hop: Reasoning across multiple conversation elements
                - Temporal: Time-based understanding and ordering
                - Commonsense: Contextual understanding and practical reasoning
                - Adversarial: Robustness under challenging conditions
                - Cross-modal: Integration of different information types
                
                Format as JSON array with question, answer, recall_type, difficulty, requires_cross_session, evidence_turns, reasoning_steps
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
        
        def process_video_to_locomo(self, video_path: str):
            """Process video into LOCOMO-compliant conversation"""
            
            print("🔄 Processing video to LOCOMO format...")
            
            # Transcribe video
            print("📝 Transcribing video...")
            result = self.whisper_model.transcribe(video_path)
            video_content = result["text"]
            
            # Create personas
            print("👥 Creating personas...")
            personas = self.create_personas_from_video(video_content)
            print(f"✅ Created {len(personas)} personas: {list(personas.keys())}")
            
            # Generate multiple sessions
            print(f"💬 Generating {self.target_sessions} conversation sessions...")
            sessions = []
            for session_num in range(self.target_sessions):
                if session_num % 5 == 0:
                    print(f"   Session {session_num + 1}/{self.target_sessions}")
                
                session = self.generate_session_conversation(session_num, personas, video_content, sessions)
                sessions.append(session)
            
            # Generate QA pairs
            print("❓ Generating QA pairs...")
            qa_pairs = self.generate_qa_pairs(sessions, personas)
            
            # Create final dataset
            dataset = {
                "metadata": {
                    "dataset_name": "LOCOMO Enhanced Dataset",
                    "source_video": video_path,
                    "processing_timestamp": datetime.now().isoformat(),
                    "total_sessions": len(sessions),
                    "total_turns": sum(len(s.get('turns', [])) for s in sessions),
                    "total_qa_pairs": len(qa_pairs),
                    "personas": list(personas.keys()),
                    "methodology": "LOCOMO Enhanced (arXiv:2402.17753)"
                },
                "personas": personas,
                "sessions": sessions,
                "qa_pairs": qa_pairs
            }
            
            print(f"✅ Generated {len(qa_pairs)} QA pairs!")
            print(f"📊 Statistics:")
            print(f"   Sessions: {len(sessions)}")
            print(f"   Total turns: {sum(len(s.get('turns', [])) for s in sessions)}")
            print(f"   QA pairs: {len(qa_pairs)}")
            print(f"   Personas: {len(personas)}")
            
            return dataset
    
    return LOCOMOEnhancedProcessor

# Usage in Colab:
# processor_class = create_locomo_enhanced_processor()
# processor = processor_class(OPENAI_API_KEY)
# processor.setup_models()
# dataset = processor.process_video_to_locomo(video_path)
