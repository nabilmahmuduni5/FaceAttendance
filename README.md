# FaceAttendance

A real-time face recognition and attendance system built with OpenCV, YuNet, InsightFace ArcFace, ONNX Runtime, and PostgreSQL.

The system is being developed as a modular face attendance pipeline that detects faces from a camera, enrolls users, aligns faces, generates 512-dimensional ArcFace embeddings, and compares embeddings for face recognition.

---

## Current Development Status

### Completed

- [x] Camera input using OpenCV
- [x] DroidCam IP camera support
- [x] YuNet face detection
- [x] Face bounding box visualization
- [x] YuNet detection confidence display
- [x] Face enrollment
- [x] Capture 30 face images per person
- [x] Minimum face-size validation
- [x] Single-face validation
- [x] Duplicate-frame prevention
- [x] Face cropping
- [x] InsightFace ArcFace `w600k_r50` model
- [x] 112×112 face alignment
- [x] 512-dimensional face embeddings
- [x] L2-normalized embeddings
- [x] Batch embedding generation
- [x] Cosine similarity testing
- [x] Same-person similarity evaluation

### In Progress

- [ ] Multiple-person recognition
- [ ] PostgreSQL database integration
- [ ] Real-time face recognition
- [ ] Attendance recording
- [ ] FAISS-based 1:N face search
- [ ] Complete attendance management system

---

# System Architecture

```text
                 Camera / DroidCam
                        |
                        v
                +----------------+
                |    OpenCV      |
                | Camera Capture |
                +----------------+
                        |
                        v
                +----------------+
                |     YuNet      |
                | Face Detection |
                +----------------+
                        |
                        v
                +----------------+
                | Face Alignment |
                |    112 x 112   |
                +----------------+
                        |
                        v
                +----------------+
                |    ArcFace     |
                |   w600k_r50    |
                +----------------+
                        |
                        v
                +----------------+
                | 512-D Embedding|
                +----------------+
                        |
                        v
                +----------------+
                |   Similarity   |
                |    Matching    |
                +----------------+
                        |
                        v
                +----------------+
                |   Attendance   |
                |    System      |
                +----------------+
