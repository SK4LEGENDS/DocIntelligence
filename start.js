const { spawn, execSync } = require("child_process");
const path = require("path");
const net = require("net");
const fs = require("fs");

// Colors for terminal logs
const colors = {
  reset: "\x1b[0m",
  cyan: "\x1b[36m",
  green: "\x1b[32m",
  yellow: "\x1b[33m",
  red: "\x1b[31m"
};

console.log(`${colors.cyan}[System] Launching Hierarchical Agentic PDF Chatbot...${colors.reset}`);
console.log(`${colors.yellow}[System] Press Ctrl+C to terminate both services.${colors.reset}\n`);

// Clean up stale Qdrant lock files before starting
const qdrantLock = path.join(__dirname, "backend", "qdrant_db", ".lock");
if (fs.existsSync(qdrantLock)) {
  try {
    fs.unlinkSync(qdrantLock);
    console.log(`${colors.yellow}[System] Removed stale Qdrant lock file.${colors.reset}`);
  } catch (e) {
    console.log(`${colors.red}[System] Could not remove Qdrant lock: ${e.message}${colors.reset}`);
  }
}

// 1. Spawn Backend (FastAPI / Uvicorn)
const backendCmd = "python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload";
console.log(`${colors.cyan}[System] Starting Backend...${colors.reset}`);
const backend = spawn(backendCmd, [], {
  cwd: __dirname,
  shell: true
});

backend.stdout.on("data", (data) => {
  const output = data.toString().trim();
  if (output) console.log(`${colors.green}[Backend]${colors.reset} ${output}`);
});

backend.stderr.on("data", (data) => {
  const output = data.toString().trim();
  if (output) console.error(`${colors.yellow}[Backend Logs]${colors.reset} ${output}`);
});

backend.on("close", (code) => {
  console.log(`${colors.red}[System] Backend process exited with code ${code}${colors.reset}`);
  process.exit(code || 0);
});

// Kill entire process tree on Windows
function killTree(pid) {
  try {
    execSync(`taskkill /F /T /PID ${pid}`, { stdio: "ignore" });
  } catch (e) {
    // Process already exited
  }
}

// Function to check if backend is listening on port 8000
function checkBackendReady(callback) {
  const client = new net.Socket();
  client.connect({ port: 8000, host: "127.0.0.1" }, () => {
    client.destroy();
    callback(true);
  });
  client.on("error", () => {
    client.destroy();
    callback(false);
  });
}

// Poll backend status
let attempts = 0;
const maxAttempts = 90; // 90 seconds max wait

function waitForBackend() {
  checkBackendReady((ready) => {
    if (ready) {
      console.log(`\n${colors.green}[System] Backend is ready! Starting Frontend...${colors.reset}`);
      startFrontend();
    } else {
      attempts++;
      if (attempts >= maxAttempts) {
        console.error(`${colors.red}[System] Backend failed to start within 30 seconds. Exiting...${colors.reset}`);
        if (backend && backend.pid) killTree(backend.pid);
        process.exit(1);
      }
      process.stdout.write(".");
      setTimeout(waitForBackend, 1000);
    }
  });
}

console.log(`${colors.cyan}[System] Waiting for backend to bind to port 8000...${colors.reset}`);
waitForBackend();

let frontend;

function startFrontend() {
  const frontendDir = path.join(__dirname, "frontend");
  const frontendCmd = "npm run dev";
  frontend = spawn(frontendCmd, [], {
    cwd: frontendDir,
    shell: true
  });

  frontend.stdout.on("data", (data) => {
    const output = data.toString().trim();
    if (output) console.log(`${colors.cyan}[Frontend]${colors.reset} ${output}`);
  });

  frontend.stderr.on("data", (data) => {
    const output = data.toString().trim();
    if (output) console.error(`${colors.red}[Frontend Error]${colors.reset} ${output}`);
  });

  frontend.on("close", (code) => {
    console.log(`${colors.red}[System] Frontend process exited with code ${code}${colors.reset}`);
  });
}

// Graceful shutdown on Ctrl+C
process.on("SIGINT", () => {
  console.log(`\n${colors.cyan}[System] Shutting down services...${colors.reset}`);
  
  if (backend && backend.pid) {
    console.log(`${colors.green}[System] Terminating Backend (PID tree: ${backend.pid})...${colors.reset}`);
    killTree(backend.pid);
  }
  
  if (frontend && frontend.pid) {
    console.log(`${colors.cyan}[System] Terminating Frontend (PID tree: ${frontend.pid})...${colors.reset}`);
    killTree(frontend.pid);
  }
  
  setTimeout(() => {
    process.exit(0);
  }, 1500);
});
