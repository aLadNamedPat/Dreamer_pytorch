import os
import numpy as np

def initialize_dmc_environment():
    """Initialize DMC environment before any imports"""
    print("Pre-configuring environment for DMC...")

    # Check what's available
    import ctypes.util
    import subprocess

    osmesa_available = ctypes.util.find_library('OSMesa') is not None

    try:
        subprocess.run(['which', 'Xvfb'], check=True, capture_output=True)
        xvfb_available = True
    except:
        xvfb_available = False

    if osmesa_available:
        print("Pre-setting OSMesa backend...")
        os.environ['MUJOCO_GL'] = 'osmesa'
        os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
        os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
    elif xvfb_available:
        print("Pre-setting Xvfb backend...")
        os.environ['MUJOCO_GL'] = 'glfw'
    else:
        print("Pre-setting EGL backend...")
        os.environ['MUJOCO_GL'] = 'egl'
        os.environ['PYOPENGL_PLATFORM'] = 'egl'

# Initialize BEFORE any dm_control imports
initialize_dmc_environment()

def check_rendering_libraries():
    """Check which rendering libraries are available"""

    try:
        import ctypes.util
        osmesa_lib = ctypes.util.find_library('OSMesa')
        if osmesa_lib:
            print(f"✓ OSMesa library found: {osmesa_lib}")
        else:
            print("✗ OSMesa library not found")
    except Exception as e:
        print(f"✗ Error checking OSMesa: {e}")

    try:
        import mujoco
        print(f"✓ MuJoCo version: {mujoco.__version__}")
    except Exception as e:
        print(f"✗ Error importing MuJoCo: {e}")

    try:
        import OpenGL
        print(f"✓ PyOpenGL version: {OpenGL.__version__}")
    except Exception as e:
        print(f"✗ Error importing PyOpenGL: {e}")

def detect_best_rendering_backend():
    """Detect the best rendering backend for this environment"""
    import ctypes.util
    import subprocess

    print("=== Detecting Best Rendering Backend ===")

    # Check for OSMesa library
    osmesa_available = ctypes.util.find_library('OSMesa') is not None
    print(f"OSMesa library: {'✓ Available' if osmesa_available else '✗ Not found'}")

    try:
        subprocess.run(['which', 'Xvfb'], check=True, capture_output=True)
        xvfb_available = True
        print("Xvfb: ✓ Available")
    except:
        xvfb_available = False
        print("Xvfb: ✗ Not found")

    # Choose backend based on availability
    if osmesa_available:
        print("→ Using OSMesa backend (software rendering)")
        return 'osmesa'
    elif xvfb_available:
        print("→ Using Xvfb + GLFW backend (virtual display)")
        return 'xvfb'
    else:
        print("→ Trying EGL backend (hardware rendering)")
        return 'egl'

def setup_rendering_environment(backend):
    """Setup environment variables for the chosen backend"""

    # Clear any existing conflicting variables
    vars_to_clear = ['DISPLAY', 'XAUTHORITY', 'MUJOCO_GL', 'PYOPENGL_PLATFORM',
                     'LIBGL_ALWAYS_SOFTWARE', 'MESA_GL_VERSION_OVERRIDE']

    for var in vars_to_clear:
        if var in os.environ:
            del os.environ[var]

    if backend == 'osmesa':
        print("Setting up OSMesa software rendering...")
        os.environ['MUJOCO_GL'] = 'osmesa'
        os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
        os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'

    elif backend == 'xvfb':
        print("Setting up Xvfb virtual display...")
        import subprocess
        import time

        # Start Xvfb
        subprocess.Popen([
            'Xvfb', ':99', '-screen', '0', '1024x768x24', '-ac', '+extension', 'GLX'
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        time.sleep(3)  # Give Xvfb time to start
        os.environ['DISPLAY'] = ':99'
        os.environ['MUJOCO_GL'] = 'glfw'

    elif backend == 'egl':
        print("Setting up EGL rendering...")
        os.environ['MUJOCO_GL'] = 'egl'
        os.environ['PYOPENGL_PLATFORM'] = 'egl'

    print(f"Environment variables set: MUJOCO_GL={os.environ.get('MUJOCO_GL')}")


def create_dmc_env_safe(domain_name="walker", task_name="walk", height=64, width=64, camera_id=0):
    """Create DMC environment with single-shot rendering setup"""

    check_rendering_libraries()

    # Detect and setup the best backend ONCE at startup
    backend = detect_best_rendering_backend()
    setup_rendering_environment(backend)

    print(f"\n=== Creating DMC Environment with {backend.upper()} backend ===")

    try:
        # Import AFTER setting up environment
        from dm_control import suite
        from dm_control.suite.wrappers import pixels

        print("Loading DMC suite environment...")
        env = suite.load(domain_name=domain_name, task_name=task_name)
        env = pixels.Wrapper(
            env,
            pixels_only=True,
            render_kwargs={'height': height, 'width': width, 'camera_id': camera_id}
        )
        time_step = env.reset()
        pixels_obs = time_step.observation['pixels']
        return env

    except Exception as e:
        print(f"\nFAILED with {backend} backend")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

        print("\n" + "="*60)
        print("RENDERING SETUP FAILED")
        print("Please install missing packages:")
        print("  sudo apt update")
        print("  sudo apt install -y libosmesa6-dev mesa-utils xvfb")
        print("="*60)

        raise RuntimeError(f"DMC environment creation failed with {backend} backend")


class CPURenderWrapper:
    """Custom wrapper for CPU-only rendering"""

    def __init__(self, env, height=64, width=64, camera_id=0):
        self._env = env
        self._height = height
        self._width = width
        self._camera_id = camera_id

    def reset(self):
        time_step = self._env.reset()
        # Add CPU-rendered pixels to observation
        pixels = self._env.physics.render(
            height=self._height,
            width=self._width,
            camera_id=self._camera_id
        )

        # Create observation dict similar to pixels wrapper
        obs = {'pixels': pixels}

        # Return modified time_step
        from dm_control.rl import control
        return control.TimeStep(
            step_type=time_step.step_type,
            reward=time_step.reward,
            discount=time_step.discount,
            observation=obs
        )

    def step(self, action):
        time_step = self._env.step(action)
        # Add CPU-rendered pixels to observation
        pixels = self._env.physics.render(
            height=self._height,
            width=self._width,
            camera_id=self._camera_id
        )

        # Create observation dict similar to pixels wrapper
        obs = {'pixels': pixels}

        # Return modified time_step
        from dm_control.rl import control
        return control.TimeStep(
            step_type=time_step.step_type,
            reward=time_step.reward,
            discount=time_step.discount,
            observation=obs
        )

    def action_spec(self):
        return self._env.action_spec()

    def observation_spec(self):
        # Mock the pixels observation spec
        return {'pixels': self._env.observation_spec()}

    def close(self):
        self._env.close()

class DMCWrapper:
    """Wrapper to make DMC environment compatible with Gymnasium interface"""
    def __init__(self, dmc_env):
        self.env = dmc_env
        self.action_spec = dmc_env.action_spec()
        self.observation_spec = dmc_env.observation_spec()

    @property
    def action_space(self):
        """Mock action space for compatibility"""
        class ActionSpace:
            def __init__(self, action_spec):
                self.shape = (len(action_spec.minimum),)
                self.low = action_spec.minimum
                self.high = action_spec.maximum

            def sample(self):
                return np.random.uniform(self.low, self.high)

        return ActionSpace(self.action_spec)

    @property
    def observation_space(self):
        """Mock observation space for compatibility"""
        class ObservationSpace:
            def __init__(self, obs_spec):
                if 'pixels' in obs_spec:
                    self.shape = obs_spec['pixels'].shape
                else:
                    # Fallback for other observation types
                    self.shape = (64, 64, 3)

        return ObservationSpace(self.observation_spec)

    def reset(self):
        time_step = self.env.reset()
        obs = time_step.observation['pixels']
        return obs, {}

    def step(self, action):
        time_step = self.env.step(action)
        obs = time_step.observation['pixels']
        reward = time_step.reward or 0.0
        terminated = time_step.last()
        truncated = False  # DMC doesn't distinguish between terminated and truncated
        info = {}
        return obs, reward, terminated, truncated, info

    def close(self):
        self.env.close()
