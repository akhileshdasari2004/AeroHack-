import streamlit as st
import cv2
import numpy as np
from PIL import Image
import kociemba
import pickle
import pandas as pd
from image_processing import detect_grid, classifiy_grid
import time

# Load the trained model
@st.cache_resource
def load_model():
    try:
        loaded_model = pickle.load(open("model.sav", 'rb'))
        return loaded_model
    except Exception as e:
        st.error(f"Model file 'model.sav' not found! Error: {str(e)}")
        return None

class RubiksCubeSolver:
    def __init__(self):
        self.model = load_model()
        # Initialize with default values matching the backend
        self.grid = []
        self.face = []
        self.solution = []
        self.green_str = "FFFFFFFFF"
        self.white_str = "UUUUUUUUU"
        self.red_str = "RRRRRRRRR"
        self.orange_str = "LLLLLLLLL"
        self.blue_str = "BBBBBBBBB"
        self.yellow_str = "DDDDDDDDD"
        
        # Initialize cube sides matching the backend
        self.green_side = [0,0,0,0,0,0,0,0,0]
        self.yellow_side = [5,5,5,5,5,5,5,5,5]
        self.blue_side = [4,4,4,4,4,4,4,4,4]
        self.orange_side = [3,3,3,3,3,3,3,3,3]
        self.white_side = [1,1,1,1,1,1,1,1,1]
        self.red_side = [2,2,2,2,2,2,2,2,2]
        
        self.solve_status = False
        self.scanned_faces = set()  # Track which faces have been scanned
    
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
        """Solve the cube using Kociemba algorithm - matching backend logic"""
        try:
            # Create cube string in the same order as backend
            cube_string = self.white_str + self.red_str + self.green_str + self.yellow_str + self.orange_str + self.blue_str
            self.solution = kociemba.solve(cube_string).split(" ")
            self.solve_status = True
            return True
        except Exception as e:
            st.error(f"Error solving cube: {str(e)}")
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
        """Reset the cube state - matching backend reset logic"""
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

def create_cube_visualization(solver):
    """Create a 2D net visualization of the cube like the original app"""
    # Colors for each face - using proper Rubik's cube colors
    colors = {
        0: "#00FF00",  # Green
        1: "#FFFFFF",  # White  
        2: "#FF0000",  # Red
        3: "#FF8C00",  # Orange
        4: "#0000FF",  # Blue
        5: "#FFFF00"   # Yellow
    }
    
    # Create the cube net layout with better styling
    html = """
    <div style="display: flex; justify-content: center; align-items: center; margin: 20px;">
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; grid-template-rows: 1fr 1fr 1fr; gap: 10px; width: 500px; height: 400px;">
    """
    
    # Define the cube net layout (like the original app)
    # Top row: empty, white, empty
    # Middle row: orange, green, red
    # Bottom row: empty, yellow, empty
    
    # White face (top center)
    html += f'<div style="grid-column: 2; grid-row: 1; display: grid; grid-template-columns: repeat(3, 1fr); grid-template-rows: repeat(3, 1fr); gap: 3px; background: #333; padding: 5px; border-radius: 5px;">'
    for i in range(9):
        color = colors.get(solver.white_side[i], "#CCCCCC")
        html += f'<div style="background: {color}; border: 2px solid #000; border-radius: 3px; min-height: 35px; box-shadow: inset 0 0 5px rgba(0,0,0,0.3);"></div>'
    html += '</div>'
    
    # Orange face (left center)
    html += f'<div style="grid-column: 1; grid-row: 2; display: grid; grid-template-columns: repeat(3, 1fr); grid-template-rows: repeat(3, 1fr); gap: 3px; background: #333; padding: 5px; border-radius: 5px;">'
    for i in range(9):
        color = colors.get(solver.orange_side[i], "#CCCCCC")
        html += f'<div style="background: {color}; border: 2px solid #000; border-radius: 3px; min-height: 35px; box-shadow: inset 0 0 5px rgba(0,0,0,0.3);"></div>'
    html += '</div>'
    
    # Green face (center)
    html += f'<div style="grid-column: 2; grid-row: 2; display: grid; grid-template-columns: repeat(3, 1fr); grid-template-rows: repeat(3, 1fr); gap: 3px; background: #333; padding: 5px; border-radius: 5px;">'
    for i in range(9):
        color = colors.get(solver.green_side[i], "#CCCCCC")
        html += f'<div style="background: {color}; border: 2px solid #000; border-radius: 3px; min-height: 35px; box-shadow: inset 0 0 5px rgba(0,0,0,0.3);"></div>'
    html += '</div>'
    
    # Red face (right center)
    html += f'<div style="grid-column: 3; grid-row: 2; display: grid; grid-template-columns: repeat(3, 1fr); grid-template-rows: repeat(3, 1fr); gap: 3px; background: #333; padding: 5px; border-radius: 5px;">'
    for i in range(9):
        color = colors.get(solver.red_side[i], "#CCCCCC")
        html += f'<div style="background: {color}; border: 2px solid #000; border-radius: 3px; min-height: 35px; box-shadow: inset 0 0 5px rgba(0,0,0,0.3);"></div>'
    html += '</div>'
    
    # Yellow face (bottom center)
    html += f'<div style="grid-column: 2; grid-row: 3; display: grid; grid-template-columns: repeat(3, 1fr); grid-template-rows: repeat(3, 1fr); gap: 3px; background: #333; padding: 5px; border-radius: 5px;">'
    for i in range(9):
        color = colors.get(solver.yellow_side[i], "#CCCCCC")
        html += f'<div style="background: {color}; border: 2px solid #000; border-radius: 3px; min-height: 35px; box-shadow: inset 0 0 5px rgba(0,0,0,0.3);"></div>'
    html += '</div>'
    
    html += """
        </div>
    </div>
    """
    
    return html

def create_face_status_display(solver):
    """Create a status display showing which faces are scanned"""
    html = """
    <div style="display: flex; justify-content: space-around; margin: 10px 0; padding: 10px; background: #f0f0f0; border-radius: 5px;">
    """
    
    faces = [
        ("White", "White" in solver.scanned_faces),
        ("Green", "Green" in solver.scanned_faces),
        ("Red", "Red" in solver.scanned_faces),
        ("Orange", "Orange" in solver.scanned_faces),
        ("Blue", "Blue" in solver.scanned_faces),
        ("Yellow", "Yellow" in solver.scanned_faces)
    ]
    
    for face_name, is_scanned in faces:
        status = "✅" if is_scanned else "❌"
        color = "#4CAF50" if is_scanned else "#f44336"
        html += f'<div style="text-align: center; padding: 5px; border-radius: 3px; background: {color}; color: white; font-weight: bold;">{status} {face_name}</div>'
    
    html += "</div>"
    return html

def main():
    st.set_page_config(
        page_title="Rubik's Cube Solver - Camera Mode",
        page_icon="🧩",
        layout="wide"
    )
    
    st.title("🧩 Rubik's Cube Solver - Camera Mode")
    st.markdown("---")
    
    # Initialize the solver
    if 'solver' not in st.session_state:
        st.session_state.solver = RubiksCubeSolver()
    
    solver = st.session_state.solver
    
    # Progress tracking
    progress = len(solver.scanned_faces) / 6
    st.progress(progress)
    st.info(f"📊 Progress: {len(solver.scanned_faces)}/6 faces scanned")
    
    # Instructions
    st.subheader("📖 How to Use")
    st.markdown("""
    ### **🎯 Simple Process:**
    
    1. **📷 Show each face to the camera**
       - Hold your cube so the camera can see one face clearly
       - Take a photo when the grid is detected (red rectangles appear)
       - The app will automatically detect which face it is
    
    2. **🔄 Repeat for all 6 faces**
       - Show each face: White, Green, Red, Orange, Blue, Yellow
       - The app tracks your progress automatically
    
    3. **🔧 Get your solution**
       - Once all faces are scanned, click "Solve"
       - Get step-by-step solving instructions
    
    ### **🎨 Face Colors Expected:**
    - 🟢 **Green** (front)
    - ⚪ **White** (top)  
    - 🔴 **Red** (right)
    - 🟠 **Orange** (left)
    - 🔵 **Blue** (back)
    - 🟡 **Yellow** (bottom)
    """)
    
    # Create two main columns
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("📊 Cube Status")
        
        # Display the cube visualization
        st.markdown(create_cube_visualization(solver), unsafe_allow_html=True)
        
        # Display face status
        st.markdown(create_face_status_display(solver), unsafe_allow_html=True)
        
        # Display current face data
        st.write("**Face Data:**")
        st.code(f"Green:  {solver.green_str}")
        st.code(f"White:  {solver.white_str}")
        st.code(f"Red:    {solver.red_str}")
        st.code(f"Orange: {solver.orange_str}")
        st.code(f"Blue:   {solver.blue_str}")
        st.code(f"Yellow: {solver.yellow_str}")
        
        # Control buttons
        col_solve, col_reset = st.columns(2)
        with col_solve:
            if st.button("🔧 Solve", type="primary", disabled=not solver.all_faces_scanned()):
                if solver.solve_cube():
                    st.success("✅ Solution generated!")
                    st.session_state.solution_ready = True
                else:
                    st.error("❌ Failed to generate solution.")
        
        with col_reset:
            if st.button("🔄 Reset"):
                solver.reset_cube()
                st.success("✅ Cube state reset!")
                if 'solution_ready' in st.session_state:
                    del st.session_state.solution_ready
                st.rerun()
    
    with col_right:
        st.subheader("📷 Camera Scanner")
        st.markdown("**Show each face of your cube to the camera**")
        
        # Tips for better scanning
        with st.expander("💡 Scanning Tips"):
            st.markdown("""
            **For best results:**
            - **Bright, even lighting** - avoid shadows
            - **Clean cube surface** - wipe off any dirt
            - **Square positioning** - hold cube so face is square to camera
            - **Good contrast** - ensure colors are clearly different
            - **Steady hands** - avoid blurry photos
            - **Show one face at a time** - don't show multiple faces
            """)
        
        # Camera input
        camera_input = st.camera_input("📸", key="camera")
        
        if camera_input is not None:
            # Convert to OpenCV format
            image = Image.open(camera_input)
            image_array = np.array(image)
            image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            # Process the image using backend functions
            processed_image, grid = detect_grid(image_cv)
            
            # Display the processed image
            st.image(processed_image, caption="Live Camera Feed with Grid Detection", use_column_width=True)
            
            if len(grid) == 9:
                st.success("✅ Grid detected! 9 squares found.")
                
                # Classify the grid using backend function
                face_string, predictions = classifiy_grid(grid)
                
                if face_string:
                    st.info(f"Detected: {face_string}")
                    
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
                            st.warning(f"⚠️ {detected_face} already scanned")
                            st.info("Please show a different face to the camera.")
                        else:
                            st.success(f"🎯 {detected_face} face detected")
                            
                            if st.button(f"✅ Save {detected_face}", key="save_btn"):
                                solver.scan_face(detected_face, face_string, predictions)
                                st.success(f"✅ {detected_face} saved! ({len(solver.scanned_faces)}/6)")
                                
                                # Auto-solve when all faces are scanned
                                if solver.all_faces_scanned():
                                    st.balloons()
                                    st.success("🎉 All faces scanned! Generating solution...")
                                    if solver.solve_cube():
                                        st.success("✅ Solution generated!")
                                        st.session_state.solution_ready = True
                                    else:
                                        st.error("❌ Failed to generate solution.")
                    else:
                        st.error("❌ Could not detect face type from center color.")
                        st.info("**Try these steps:**")
                        st.markdown("""
                        1. **Improve lighting** - use brighter, more even light
                        2. **Clean the cube** - wipe off any dirt or smudges
                        3. **Reposition** - hold the cube more squarely to the camera
                        4. **Check colors** - ensure the center color is clearly visible
                        5. **Try again** - take another photo
                        """)
            else:
                st.warning(f"⚠️ Grid not detected. Found {len(grid)} squares.")
                st.info("**Reposition the cube:**")
                st.markdown("""
                1. **Hold the cube squarely** to the camera
                2. **Ensure all 9 squares** are clearly visible
                3. **Avoid shadows** and reflections
                4. **Try different angles** if needed
                """)
    
    # Solution section
    if hasattr(st.session_state, 'solution_ready') and st.session_state.solution_ready:
        st.markdown("---")
        st.subheader("📋 Solution Steps")
        
        solution_text = solver.get_solution_steps()
        st.text_area("Step-by-Step Solution:", solution_text, height=200)
        
        # Download solution
        st.download_button(
            label="📥 Download Solution",
            data=solution_text,
            file_name="rubiks_solution.txt",
            mime="text/plain"
        )
        
        # Show solution statistics
        if solver.solution:
            st.info(f"📊 Solution Statistics:")
            st.write(f"- Total moves: {len(solver.solution)}")
            st.write(f"- Algorithm used: Kociemba")
            st.write(f"- Expected solve time: ~{len(solver.solution) * 2} seconds")
            
            # Move notation guide
            st.subheader("📖 Move Notation Guide")
            st.markdown("""
            **Basic Moves:**
            - `R` = Right face clockwise
            - `R'` = Right face counter-clockwise  
            - `R2` = Right face 180 degrees
            - `U` = Up face clockwise
            - `L` = Left face clockwise
            - `D` = Down face clockwise
            - `F` = Front face clockwise
            - `B` = Back face clockwise
            """)
    
    # Final instructions
    st.markdown("---")
    st.subheader("💡 Tips for Best Results")
    st.markdown("""
    - **Good lighting** - bright, even lighting
    - **Clean cube** - make sure colors are clearly visible
    - **Square positioning** - hold the cube so the face is square to the camera
    - **One face at a time** - show each face separately
    - **Steady hands** - avoid blurry photos
    - **Check progress** - see which faces are already scanned
    """)
    
    st.markdown("---")
    st.markdown("**🎉 That's it! Just show each face to the camera and get your solution!**")

if __name__ == "__main__":
    main() 