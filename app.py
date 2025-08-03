from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import kociemba
import pickle
import base64
import io
import os
from image_processing import detect_grid, classifiy_grid

app = Flask(__name__)
CORS(app)

# Load the trained model
def load_model():
    try:
        loaded_model = pickle.load(open("model.sav", 'rb'))
        return loaded_model
    except Exception as e:
        print(f"Model file 'model.sav' not found! Error: {str(e)}")
        return None

class RubiksCubeSolver:
    def __init__(self):
        self.model = load_model()
        # Initialize with default values
        self.grid = []
        self.face = []
        self.solution = []
        self.green_str = "FFFFFFFFF"
        self.white_str = "UUUUUUUUU"
        self.red_str = "RRRRRRRRR"
        self.orange_str = "LLLLLLLLL"
        self.blue_str = "BBBBBBBBB"
        self.yellow_str = "DDDDDDDDD"
        
        # Initialize cube sides
        self.green_side = [0,0,0,0,0,0,0,0,0]
        self.yellow_side = [5,5,5,5,5,5,5,5,5]
        self.blue_side = [4,4,4,4,4,4,4,4,4]
        self.orange_side = [3,3,3,3,3,3,3,3,3]
        self.white_side = [1,1,1,1,1,1,1,1,1]
        self.red_side = [2,2,2,2,2,2,2,2,2]
        
        self.solve_status = False
        self.scanned_faces = set()
    
    def scan_face(self, face_name, face_data, side_data):
        """Scan a face and update the corresponding string and side data"""
        if face_name == "Green":
            self.green_str = face_data
            self.green_side = side_data
            self.scanned_faces.add("Green")
        elif face_name == "White":
            self.white_str = face_data
            self.white_side = side_data
            self.scanned_faces.add("White")
        elif face_name == "Red":
            self.red_str = face_data
            self.red_side = side_data
            self.scanned_faces.add("Red")
        elif face_name == "Orange":
            self.orange_str = face_data
            self.orange_side = side_data
            self.scanned_faces.add("Orange")
        elif face_name == "Blue":
            self.blue_str = face_data
            self.blue_side = side_data
            self.scanned_faces.add("Blue")
        elif face_name == "Yellow":
            self.yellow_str = face_data
            self.yellow_side = side_data
            self.scanned_faces.add("Yellow")
    
    def solve_cube(self):
        """Solve the cube using Kociemba algorithm"""
        try:
            # Create cube string in the same order as backend
            cube_string = self.white_str + self.red_str + self.green_str + self.yellow_str + self.orange_str + self.blue_str
            self.solution = kociemba.solve(cube_string).split(" ")
            self.solve_status = True
            return True
        except Exception as e:
            print(f"Error solving cube: {str(e)}")
            return False
    
    def get_solution_steps(self):
        """Get the solution steps as a formatted string"""
        if not self.solution:
            return "No solution available. Please scan all faces first."
        
        steps = []
        for i, step in enumerate(self.solution, 1):
            steps.append(f"{i}. {step}")
        return "\n".join(steps)
    
    def reset_cube(self):
        """Reset the cube state"""
        self.solve_status = False
        self.grid = []
        self.face = []
        self.solution = []
        self.scanned_faces.clear()
        self.green_str = "FFFFFFFFF"
        self.white_str = "UUUUUUUUU"
        self.red_str = "RRRRRRRRR"
        self.orange_str = "LLLLLLLLL"
        self.blue_str = "BBBBBBBBB"
        self.yellow_str = "DDDDDDDDD"
        self.green_side = [0,0,0,0,0,0,0,0,0]
        self.yellow_side = [5,5,5,5,5,5,5,5,5]
        self.blue_side = [4,4,4,4,4,4,4,4,4]
        self.orange_side = [3,3,3,3,3,3,3,3,3]
        self.white_side = [1,1,1,1,1,1,1,1,1]
        self.red_side = [2,2,2,2,2,2,2,2,2]
    
    def all_faces_scanned(self):
        """Check if all 6 faces have been scanned"""
        return len(self.scanned_faces) == 6

# Global solver instance
solver = RubiksCubeSolver()

@app.route('/api/process-image', methods=['POST'])
def process_image():
    """Process uploaded image and detect cube face"""
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        # Remove data URL prefix
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image)
        image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Process the image using backend functions
        processed_image, grid = detect_grid(image_cv)
        
        # Convert processed image back to base64 for frontend
        _, buffer = cv2.imencode('.jpg', processed_image)
        processed_image_b64 = base64.b64encode(buffer).decode('utf-8')
        
        response = {
            'success': True,
            'processed_image': f'data:image/jpeg;base64,{processed_image_b64}',
            'grid_detected': len(grid) == 9,
            'grid_count': len(grid)
        }
        
        if len(grid) == 9:
            # Classify the grid using backend function
            face_string, predictions = classifiy_grid(grid)
            
            if face_string:
                # Determine which face this is based on center color
                center_color = predictions[4] if len(predictions) > 4 else None
                face_mapping = {
                    0: "Green",
                    1: "White", 
                    2: "Red",
                    3: "Orange",
                    4: "Blue",
                    5: "Yellow"
                }
                
                detected_face = face_mapping.get(center_color, "Unknown")
                
                if detected_face != "Unknown":
                    if detected_face in solver.scanned_faces:
                        response['message'] = f"{detected_face} already scanned"
                        response['status'] = 'already_scanned'
                    else:
                        response['detected_face'] = detected_face
                        response['face_string'] = face_string
                        response['predictions'] = predictions.tolist()
                        response['status'] = 'new_face'
                else:
                    response['message'] = "Could not detect face type"
                    response['status'] = 'unknown_face'
            else:
                response['message'] = "Could not classify grid"
                response['status'] = 'classification_failed'
        else:
            response['message'] = f"Grid not detected. Found {len(grid)} squares."
            response['status'] = 'no_grid'
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/save-face', methods=['POST'])
def save_face():
    """Save a detected face to the solver"""
    try:
        data = request.get_json()
        face_name = data.get('face_name')
        face_string = data.get('face_string')
        predictions = data.get('predictions')
        
        solver.scan_face(face_name, face_string, predictions)
        
        # Auto-solve when all faces are scanned
        if solver.all_faces_scanned():
            solver.solve_cube()
        
        return jsonify({
            'success': True,
            'scanned_faces': list(solver.scanned_faces),
            'progress': len(solver.scanned_faces) / 6,
            'all_faces_scanned': solver.all_faces_scanned(),
            'solution_ready': solver.solve_status
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/get-solution', methods=['GET'])
def get_solution():
    """Get the solution steps"""
    try:
        if solver.solve_status:
            solution_steps = solver.get_solution_steps()
            return jsonify({
                'success': True,
                'solution': solution_steps,
                'moves': solver.solution,
                'move_count': len(solver.solution) if solver.solution else 0
            })
        else:
            return jsonify({
                'success': False,
                'message': 'No solution available. Please scan all faces first.'
            })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def reset_cube():
    """Reset the cube state"""
    try:
        solver.reset_cube()
        return jsonify({
            'success': True,
            'message': 'Cube state reset successfully'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current solver status"""
    try:
        return jsonify({
            'success': True,
            'scanned_faces': list(solver.scanned_faces),
            'progress': len(solver.scanned_faces) / 6,
            'all_faces_scanned': solver.all_faces_scanned(),
            'solution_ready': solver.solve_status,
            'cube_state': {
                'green_side': solver.green_side,
                'white_side': solver.white_side,
                'red_side': solver.red_side,
                'orange_side': solver.orange_side,
                'blue_side': solver.blue_side,
                'yellow_side': solver.yellow_side
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/')
def home():
    return jsonify({
        'message': 'Rubik\'s Cube Solver API',
        'status': 'running',
        'endpoints': [
            '/api/process-image',
            '/api/save-face', 
            '/api/get-solution',
            '/api/reset',
            '/api/status'
        ]
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) 