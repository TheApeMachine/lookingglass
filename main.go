package main

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/fatih/color"
)

// Module represents a code module with its own dependencies and venv
type Module struct {
	Path         string
	Requirements string
	Env          []string // Build/Setup env vars
	Python       string   // Python version/command to use for venv creation
	setupDone    bool
	setupMu      sync.Mutex
}

// Service represents a running process belonging to a module
type Service struct {
	Name    string
	Module  *Module
	Command string // Command relative to module path (or absolute)
	Args    []string
	Env     []string // Runtime env vars
	Color   func(a ...interface{}) string
}

func main() {
	// Define Modules (directories with venvs)
	gpuModule := &Module{
		Path:         "gpu-service",
		Requirements: "requirements.txt",
		Python:       "python3.12",
		Env:          []string{"CMAKE_ARGS=-DDLIB_PNG_SUPPORT=OFF"},
	}

	graphModule := &Module{
		Path:         "graph-service",
		Requirements: "requirements.txt",
		Python:       "python3.12",
	}

	lookupModule := &Module{
		Path:         "lookup",
		Requirements: "requirements.txt",
		Python:       "python3.12",
	}

	crawlerModule := &Module{
		Path:         "crawler",
		Requirements: "requirements.txt",
		Python:       "python3.12",
	}

	// Define Services
	services := []Service{
		{
			Name:    "GPU-Worker",
			Module:  gpuModule,
			Command: "venv/bin/rq",
			Args:    []string{"worker", "--url", "redis://localhost:6379", "--with-scheduler", "facial_recognition"},
			Env: []string{
				"MINIO_ENDPOINT=localhost:9000",
				"MINIO_USER=miniouser",
				"MINIO_PASSWORD=miniopassword",
				"MINIO_BUCKET=scraped",
				"REDIS_HOST=localhost",
				"REDIS_PORT=6379",
				"OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES",
				"PYTORCH_ENABLE_MPS_FALLBACK=1",
			},
			Color: color.New(color.FgCyan).SprintFunc(),
		},
		{
			Name:    "GPU-API",
			Module:  gpuModule,
			Command: "venv/bin/python",
			Args:    []string{"main.py"},
			Env: []string{
				"MINIO_ENDPOINT=localhost:9000",
				"MINIO_USER=miniouser",
				"MINIO_PASSWORD=miniopassword",
				"MINIO_BUCKET=scraped",
				"REDIS_HOST=localhost",
				"REDIS_PORT=6379",
				"OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES",
				"PYTORCH_ENABLE_MPS_FALLBACK=1",
			},
			Color: color.New(color.FgBlue).SprintFunc(),
		},
		{
			Name:    "Graph-Worker",
			Module:  graphModule,
			Command: "venv/bin/python",
			Args:    []string{"page_worker.py"},
			Env: []string{
				"NEO4J_HOST=localhost",
				"NEO4J_USER=neo4j",
				"NEO4J_PASSWORD=password",
				"MINIO_ENDPOINT=localhost:9000",
				"MINIO_USER=miniouser",
				"MINIO_PASSWORD=miniopassword",
				"MINIO_BUCKET=scraped",
			},
			Color: color.New(color.FgGreen).SprintFunc(),
		},
		{
			Name:    "Graph-Analytics",
			Module:  graphModule,
			Command: "venv/bin/python",
			Args:    []string{"analyze.py"},
			Env: []string{
				"NEO4J_HOST=localhost",
				"NEO4J_USER=neo4j",
				"NEO4J_PASSWORD=password",
				"MINIO_ENDPOINT=localhost:9000",
				"MINIO_USER=miniouser",
				"MINIO_PASSWORD=miniopassword",
				"MINIO_BUCKET=scraped",
			},
			Color: color.New(color.FgMagenta).SprintFunc(),
		},
		{
			Name:    "Graph-Curator",
			Module:  graphModule,
			Command: "venv/bin/python",
			Args:    []string{"curator.py"},
			Env: []string{
				"NEO4J_HOST=localhost",
				"NEO4J_USER=neo4j",
				"NEO4J_PASSWORD=password",
				"OPENAI_API_KEY=lm-studio",
				"OPENAI_BASE_URL=http://localhost:1234/v1",
				"OPENAI_MODEL=openai/gpt-oss-20b",
			},
			Color: color.New(color.FgCyan).SprintFunc(),
		},
		{
			Name:    "Graph-Enricher",
			Module:  graphModule,
			Command: "venv/bin/python",
			Args:    []string{"enricher.py"},
			Env: []string{
				"NEO4J_HOST=localhost",
				"NEO4J_USER=neo4j",
				"NEO4J_PASSWORD=password",
				"OPENAI_API_KEY=lm-studio",
				"OPENAI_BASE_URL=http://localhost:1234/v1",
				"OPENAI_MODEL=openai/gpt-oss-20b",
			},
			Color: color.New(color.FgYellow).SprintFunc(),
		},
		{
			Name:    "Lookup-App",
			Module:  lookupModule,
			Command: "venv/bin/python",
			Args:    []string{"web_app.py"},
			Env: []string{
				"GPU_WORKER_URL=http://localhost:5001/lookup",
				"MINIO_ENDPOINT=localhost:9000",
				"MINIO_USER=miniouser",
				"MINIO_PASSWORD=miniopassword",
				"MINIO_BUCKET=scraped",
			},
			Color: color.New(color.FgYellow).SprintFunc(),
		},
		{
			Name:    "Media-Crawler",
			Module:  crawlerModule,
			Command: "venv/bin/scrapy",
			Args:    []string{"crawl", "media_spider"},
			Env: []string{
				"CRAWL_MODE=image",
				"START_URL=https://fanfactory.nl",
				"MINIO_ENDPOINT=localhost:9000",
				"MINIO_USER=miniouser",
				"MINIO_PASSWORD=miniopassword",
				"MINIO_BUCKET=scraped",
			},
			Color: color.New(color.FgRed).SprintFunc(),
		},
	}

	// 1. Setup Modules (Venv + Deps)
	fmt.Println("Setting up modules...")
	for _, svc := range services {
		if err := setupModule(svc.Module); err != nil {
			fmt.Printf("Error setting up module for %s: %v\n", svc.Name, err)
			os.Exit(1)
		}
	}
	fmt.Println("All modules set up successfully.")

	// Ensure Playwright browsers are installed for crawler services
	for _, svc := range services {
		if svc.Module.Path == "crawler" {
			if err := ensurePlaywrightBrowsers(svc.Module); err != nil {
				fmt.Printf("Warning: Failed to ensure Playwright browsers for %s: %v\n", svc.Name, err)
			}
		}
	}

	// 2. Run Services
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var wg sync.WaitGroup

	// Handle interrupt signals
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		<-sigChan
		fmt.Println("\nReceived shutdown signal, stopping services...")
		cancel()
	}()

	fmt.Println("Starting services...")

	for _, svc := range services {
		wg.Add(1)
		go runService(ctx, &wg, svc)
	}

	wg.Wait()
	fmt.Println("All services stopped.")
}

func setupModule(m *Module) error {
	m.setupMu.Lock()
	defer m.setupMu.Unlock()

	if m.setupDone {
		return nil
	}

	fmt.Printf("Setting up module: %s...\n", m.Path)

	venvPath := filepath.Join(m.Path, "venv")
	pythonPath := filepath.Join(venvPath, "bin", "python")

	// Check if venv exists and is valid (python executable exists and is valid)
	// os.Stat follows symlinks, so this checks if the target exists too
	if _, err := os.Stat(pythonPath); os.IsNotExist(err) {
		fmt.Printf("Creating venv in %s...\n", venvPath)
		// Clean up any partial/broken venv
		os.RemoveAll(venvPath)

		cmd := exec.Command(m.Python, "-m", "venv", venvPath)
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			return fmt.Errorf("failed to create venv: %w", err)
		}
	}

	// Resolve absolute path to python executable to ensure it works regardless of cmd.Dir
	absVenvPath, _ := filepath.Abs(venvPath)
	pythonCmd := filepath.Join(absVenvPath, "bin", "python")

	// Install requirements
	reqPath := filepath.Join(m.Path, m.Requirements)
	if _, err := os.Stat(reqPath); err == nil {
		fmt.Printf("Installing requirements from %s...\n", reqPath)
		// Use python -m pip instead of pip executable directly to avoid shebang issues

		cmd := exec.Command(pythonCmd, "-m", "pip", "install", "-r", m.Requirements)
		cmd.Dir = m.Path

		// Set up environment variables to mimic activating the venv
		cmd.Env = os.Environ()
		cmd.Env = append(cmd.Env, fmt.Sprintf("VIRTUAL_ENV=%s", absVenvPath))
		cmd.Env = append(cmd.Env, fmt.Sprintf("PATH=%s:%s", filepath.Join(absVenvPath, "bin"), os.Getenv("PATH")))
		cmd.Env = append(cmd.Env, m.Env...)

		// Capture output for error reporting, but stream slightly to show progress if needed
		// For now, just inheriting stdout/stderr is easiest for the user to see build progress
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr

		if err := cmd.Run(); err != nil {
			return fmt.Errorf("failed to install requirements: %w", err)
		}
	}

	// Install Playwright browsers for crawler module (always run, even if venv exists)
	if m.Path == "crawler" {
		fmt.Printf("Installing Playwright browsers for %s...\n", m.Path)
		// Install chromium (includes headless_shell)
		cmd := exec.Command(pythonCmd, "-m", "playwright", "install", "chromium")
		cmd.Dir = m.Path
		cmd.Env = os.Environ()
		cmd.Env = append(cmd.Env, fmt.Sprintf("VIRTUAL_ENV=%s", absVenvPath))
		cmd.Env = append(cmd.Env, fmt.Sprintf("PATH=%s:%s", filepath.Join(absVenvPath, "bin"), os.Getenv("PATH")))
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			fmt.Printf("Warning: Playwright browser installation had issues: %v\n", err)
		}

		// Install system dependencies
		cmd = exec.Command(pythonCmd, "-m", "playwright", "install-deps")
		cmd.Dir = m.Path
		cmd.Env = os.Environ()
		cmd.Env = append(cmd.Env, fmt.Sprintf("VIRTUAL_ENV=%s", absVenvPath))
		cmd.Env = append(cmd.Env, fmt.Sprintf("PATH=%s:%s", filepath.Join(absVenvPath, "bin"), os.Getenv("PATH")))
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			fmt.Printf("Warning: Playwright deps installation had issues: %v\n", err)
		}
	}

	m.setupDone = true
	return nil
}

func ensurePlaywrightBrowsers(m *Module) error {
	if m.Path != "crawler" {
		return nil
	}

	venvPath := filepath.Join(m.Path, "venv")
	pythonPath := filepath.Join(venvPath, "bin", "python")

	// Check if venv exists
	if _, err := os.Stat(pythonPath); os.IsNotExist(err) {
		return fmt.Errorf("venv does not exist for %s", m.Path)
	}

	absVenvPath, _ := filepath.Abs(venvPath)
	pythonCmd := filepath.Join(absVenvPath, "bin", "python")

	fmt.Printf("Ensuring Playwright browsers are installed for %s...\n", m.Path)

	// Install chromium (includes headless_shell)
	cmd := exec.Command(pythonCmd, "-m", "playwright", "install", "chromium")
	cmd.Dir = m.Path
	cmd.Env = os.Environ()
	cmd.Env = append(cmd.Env, fmt.Sprintf("VIRTUAL_ENV=%s", absVenvPath))
	cmd.Env = append(cmd.Env, fmt.Sprintf("PATH=%s:%s", filepath.Join(absVenvPath, "bin"), os.Getenv("PATH")))
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to install playwright browsers: %w", err)
	}

	return nil
}

func runService(ctx context.Context, wg *sync.WaitGroup, svc Service) {
	defer wg.Done()

	// Resolve absolute path for command if needed, or use relative to module path
	// We run the command inside the module path
	cmdPath := svc.Command
	// If the command is inside venv/bin, it's relative to module path

	cmd := exec.CommandContext(ctx, cmdPath, svc.Args...)
	cmd.Dir = svc.Module.Path

	// Setup Environment
	cmd.Env = os.Environ()
	cmd.Env = append(cmd.Env, svc.Env...)
	// Also add venv/bin to PATH for convenience
	venvBin := filepath.Join(svc.Module.Path, "venv", "bin")
	absVenvBin, _ := filepath.Abs(venvBin)
	cmd.Env = append(cmd.Env, fmt.Sprintf("PATH=%s:%s", absVenvBin, os.Getenv("PATH")))

	// Ensure the process group is killed on shutdown
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		fmt.Printf("Error creating stdout pipe for %s: %v\n", svc.Name, err)
		return
	}

	stderr, err := cmd.StderrPipe()
	if err != nil {
		fmt.Printf("Error creating stderr pipe for %s: %v\n", svc.Name, err)
		return
	}

	if err := cmd.Start(); err != nil {
		fmt.Printf("Error starting %s: %v\n", svc.Name, err)
		return
	}

	// Stream output
	var outWg sync.WaitGroup
	outWg.Add(2)

	go streamOutput(stdout, svc.Name, svc.Color, &outWg)
	go streamOutput(stderr, svc.Name, svc.Color, &outWg)

	// Wait for the command to finish
	errChan := make(chan error, 1)
	go func() {
		outWg.Wait()
		errChan <- cmd.Wait()
	}()

	select {
	case <-ctx.Done():
		// Context cancelled, kill the process group
		if cmd.Process != nil {
			// Send SIGKILL to the process group (negative PID)
			syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
		}
	case err := <-errChan:
		if err != nil {
			// Check if it was just killed by us
			if ctx.Err() == nil {
				fmt.Printf("%s exited with error: %v\n", svc.Name, err)
			}
		} else {
			fmt.Printf("%s exited successfully\n", svc.Name)
		}
	}
}

func streamOutput(r io.Reader, name string, colorFunc func(a ...interface{}) string, wg *sync.WaitGroup) {
	defer wg.Done()
	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		timestamp := time.Now().Format("15:04:05")
		// Calculate padding for alignment (assuming max name length ~15)
		padding := strings.Repeat(" ", 15-len(name))
		if len(name) > 15 {
			padding = ""
		}

		prefix := fmt.Sprintf("[%s] %s%s | ", timestamp, name, padding)
		fmt.Printf("%s%s\n", colorFunc(prefix), scanner.Text())
	}
}
