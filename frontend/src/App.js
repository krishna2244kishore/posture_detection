import React, { useState, useRef, useEffect } from 'react';
import './App.css';
import io from 'socket.io-client';

function App() {
  const [videoFile, setVideoFile] = useState(null);
  const [videoURL, setVideoURL] = useState('');
  const [feedback, setFeedback] = useState([]);
  const [repCount, setRepCount] = useState(0); // Add this state
  const [analysisDone, setAnalysisDone] = useState(false); // Track if analysis has been performed
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Webcam states
  const [recording, setRecording] = useState(false);
  const [webcamStream, setWebcamStream] = useState(null);
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const [recordedChunks, setRecordedChunks] = useState([]);
  const videoRef = useRef();
  const webcamRef = useRef();
  const recordedChunksRef = useRef([]); // <-- Add this line

  const [realtime, setRealtime] = useState(false);
  const [realtimeFeedback, setRealtimeFeedback] = useState([]);
  const intervalRef = useRef();

  const [mode, setMode] = useState('squat'); // Add this state
  const [recordedFeedback, setRecordedFeedback] = useState([]); // Store feedback timeline for recorded video
  const [liveUploadFeedback, setLiveUploadFeedback] = useState([]); // For live feedback during upload
  const [uploading, setUploading] = useState(false); // Track upload state

  const socket = io('http://localhost:5000'); // Connect to backend Socket.IO
  const apiUrl = process.env.REACT_APP_API_URL;

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setVideoFile(file);
    setVideoURL(file ? URL.createObjectURL(file) : '');
    setFeedback([]);
    setError('');
    setAnalysisDone(false);
  };

  const handleUpload = async (file = videoFile) => {
    if (!file) {
      setError('Please select or record a video file.');
      return;
    }
    setLoading(true);
    setUploading(true);
    setError('');
    setFeedback([]);
    setLiveUploadFeedback([]);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('mode', mode); // Add this line
    formData.append('sid', socket.id); // Send Socket.IO session id
    console.log('Uploading file:', file);
    console.log('FormData:', [...formData.entries()]);
    try {
      const response = await fetch(`${apiUrl}/upload`, {
        method: 'POST',
        body: formData,
      });
      let data;
      const text = await response.text();
      try {
        data = JSON.parse(text);
      } catch (e) {
        throw new Error('Server returned invalid JSON or empty response.');
      }
      if (!response.ok) {
        throw new Error((data && data.error) || 'Upload failed');
      }
      setFeedback(data.results.frames);
      setRepCount(data.results.rep_count);
      setAnalysisDone(true);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
      setUploading(false);
    }
  };

  // Webcam logic
  const startWebcam = async () => {
    setError('');
    setFeedback([]);
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setError('Webcam not supported in this browser.');
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      setWebcamStream(stream);
      if (webcamRef.current) {
        webcamRef.current.srcObject = stream;
      }
    } catch (err) {
      setError('Could not access webcam.');
    }
  };

  const stopWebcam = () => {
    if (webcamStream) {
      webcamStream.getTracks().forEach((track) => track.stop());
      setWebcamStream(null);
    }
  };

  // Modified startRecording to also start real-time analysis
  const startRecording = () => {
    if (!webcamStream) return;
    recordedChunksRef.current = [];
    setRecordedFeedback([]);
    console.log('Recording started');
    let options = { mimeType: 'video/webm;codecs=vp8,opus' };
    if (!MediaRecorder.isTypeSupported(options.mimeType)) {
      options = { mimeType: 'video/webm' };
    }
    const recorder = new window.MediaRecorder(webcamStream, options);
    setMediaRecorder(recorder);
    let startTime = Date.now();
    recorder.ondataavailable = (e) => {
      console.log('ondataavailable called, size:', e.data.size);
      if (e.data.size > 0) {
        recordedChunksRef.current.push(e.data);
        console.log('Recorded chunks length:', recordedChunksRef.current.length);
      }
    };
    recorder.onstop = () => {
      console.log('Recording stopped');
      const duration = (Date.now() - startTime) / 1000;
      if (duration < 1) {
        alert('Recording was too short. Please record for at least 2 seconds.');
      }
      const blob = new Blob(recordedChunksRef.current, { type: recorder.mimeType });
      console.log('Blob size:', blob.size);
      const file = new File([blob], 'webcam_recording.webm', { type: recorder.mimeType });
      setVideoFile(file);
      const url = URL.createObjectURL(blob);
      setVideoURL(url);
      console.log('Video URL:', url);
      setRecording(false);
      stopWebcam();
      stopRealtimeAnalysis(); // Stop feedback collection
    };
    recorder.start();
    setRecording(true);
    startRealtimeAnalysis(true); // Start collecting feedback timeline
    // Save recorder to ref for later use
    mediaRecorderRef.current = recorder;
  };

  // Ensure requestData is called before stopping
  const mediaRecorderRef = useRef();
  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.requestData();
      mediaRecorderRef.current.stop();
    }
  };

  const handleWebcamButton = () => {
    if (webcamStream) {
      stopWebcam();
    } else {
      startWebcam();
    }
  };

  const handleAnalyzeRecorded = () => {
    if (videoFile) {
      handleUpload(videoFile);
    }
  };

  // Real-time analysis function (modified for recording)
  const startRealtimeAnalysis = (recordingMode = false) => {
    setRealtime(true);
    setRealtimeFeedback([]);
    if (recordingMode) setRecordedFeedback([]); // Reset feedback timeline if recording
    intervalRef.current = setInterval(async () => {
      if (!webcamRef.current) return;
      const video = webcamRef.current;
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
      const image = canvas.toDataURL('image/jpeg').split(',')[1];
      try {
        const res = await fetch(`${apiUrl}/analyze_frame`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image }),
        });
        const data = await res.json();
        setRealtimeFeedback(data.feedback);
        if (recordingMode) {
          setRecordedFeedback(prev => [...prev, data.feedback]);
        }
      } catch (err) {
        setRealtimeFeedback(['Error analyzing frame']);
        if (recordingMode) {
          setRecordedFeedback(prev => [...prev, ['Error analyzing frame']]);
        }
      }
    }, 300); // every 300ms
  };

  const stopRealtimeAnalysis = () => {
    setRealtime(false);
    setRealtimeFeedback([]);
    if (intervalRef.current) clearInterval(intervalRef.current);
  };

  useEffect(() => {
    if (!realtime) {
      if (intervalRef.current) clearInterval(intervalRef.current);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [realtime]);

  useEffect(() => {
    // Listen for live feedback from backend
    socket.on('video_frame_feedback', (data) => {
      setLiveUploadFeedback((prev) => [...prev, data]);
    });
    return () => {
      socket.off('video_frame_feedback');
    };
  }, []);

  // Calculate rep count for recorded feedback (frames with no feedback or only 'Good posture')
  const recordedRepCount = recordedFeedback.filter(f => !f || f.length === 0 || (f.length === 1 && f[0] === 'Good posture')).length;

  return (
    <div className="App">
      <h1>Posture Detection App</h1>
      <label>
        Select Exercise:
        <select value={mode} onChange={e => setMode(e.target.value)}>
          <option value="squat">Squat</option>
          <option value="pushup">Pushup</option>
          <option value="desk">Desk Posture</option>
        </select>
      </label>
      <input type="file" accept="video/*" onChange={handleFileChange} />
      <button onClick={() => handleUpload()} disabled={loading || (!videoFile && recordedChunks.length === 0)}>
        {loading ? 'Uploading...' : 'Upload & Analyze'}
      </button>
      <div style={{ margin: '20px 0' }}>
        <button onClick={handleWebcamButton}>
          {webcamStream ? 'Stop Webcam' : 'Start Webcam'}
        </button>
        {webcamStream && !recording && !realtime && (
          <button onClick={startRecording} style={{ marginLeft: 10 }}>
            Start Recording
          </button>
        )}
        {recording && (
          <button onClick={stopRecording} style={{ marginLeft: 10, color: 'red' }}>
            Stop Recording
          </button>
        )}
        {webcamStream && !realtime && (
          <button onClick={startRealtimeAnalysis} style={{ marginLeft: 10, color: 'green' }}>
            Start Real-Time Analysis
          </button>
        )}
        {realtime && (
          <button onClick={stopRealtimeAnalysis} style={{ marginLeft: 10, color: 'orange' }}>
            Stop Real-Time Analysis
          </button>
        )}
      </div>
      {error && <div className="error">{error}</div>}
      {webcamStream && (
        <div className="video-container">
          <video ref={webcamRef} autoPlay width="480" />
        </div>
      )}
      {realtime && (
        <div className="feedback">
          <h2>Live Feedback</h2>
          <ul>
            {realtimeFeedback.map((f, idx) => (
              <li key={idx}>{f}</li>
            ))}
          </ul>
        </div>
      )}
      {uploading && liveUploadFeedback.length > 0 && (
        <div className="live-upload-feedback">
          <h2>Live Feedback (Upload in Progress)</h2>
          <ul>
            {liveUploadFeedback.map((f, idx) => (
              <li key={idx}>
                <strong>Frame {f.frame}:</strong>
                {f.feedback.length > 0
                  ? f.feedback.map((msg, i) => (
                      <span
                        key={i}
                        className={`badge ${msg === 'Good posture' ? 'good' : 'error'}`}
                      >
                        {msg}
                      </span>
                    ))
                  : <span className="badge good">Good posture</span>}
              </li>
            ))}
          </ul>
        </div>
      )}
      {videoURL && !webcamStream && (
        <div className="video-container">
          <video ref={videoRef} src={videoURL} controls width="480" />
          {recordedChunks.length > 0 && (
            <button onClick={handleAnalyzeRecorded} style={{ display: 'block', margin: '10px auto' }}>
              Analyze Recorded Video
            </button>
          )}
        </div>
      )}
      {analysisDone && (
        <div className="rep-count">
          <h2>Repetitions Counted: {repCount}</h2>
        </div>
      )}
      {feedback.length > 0 && (
        <div className="feedback">
          <h2>Feedback per Frame</h2>
          <ul>
            {feedback.map((f) => (
              <li key={f.frame}>
                <strong>Frame {f.frame}:</strong>
                {f.feedback.length > 0
                  ? f.feedback.map((msg, i) => (
                      <span
                        key={i}
                        className={`badge ${msg === 'Good posture' ? 'good' : 'error'}`}
                      >
                        {msg}
                      </span>
                    ))
                  : <span className="badge good">Good posture</span>}
              </li>
            ))}
          </ul>
        </div>
      )}
      {videoURL && recordedFeedback.length > 0 && !webcamStream && (
        <div className="feedback-timeline">
          <h2>Feedback Timeline (Recorded)</h2>
          <div className="rep-count">
            <h2>Repetitions Counted (Recorded): {recordedRepCount}</h2>
          </div>
          <ul>
            {recordedFeedback.map((f, idx) => (
              <li key={idx}>
                <strong>Frame {idx}:</strong>
                {f.length > 0
                  ? f.map((msg, i) => (
                      <span
                        key={i}
                        className={`badge ${msg === 'Good posture' ? 'good' : 'error'}`}
                      >
                        {msg}
                      </span>
                    ))
                  : <span className="badge good">Good posture</span>}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;
