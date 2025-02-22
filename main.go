package main

import (
	"fmt"
	"image/color"
	"log"
	"math"
	"math/rand"
	"runtime"
	"sync"
	"time"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
)

const (
	Width        = 800         // Screen width in pixels
	Height       = 600         // Screen height in pixels
	GridSize     = 50          // Size of grid cells for spatial partitioning
	MinDistance  = 10.0        // Minimum distance for collision and force
	K            = 1.0         // Force constant for repulsion
	Delta        = 0.01        // Time step for simulation
	NumParticles = 2500        // Number of particles
	ParticleSize = MinDistance // Size of particle for rendering (square side length)
)

// Particle represents a single particle in the Verlet simulation
type Particle struct {
	Pos     [2]float64 // Current position [x, y]
	PrevPos [2]float64 // Previous position [x, y]
	Accel   [2]float64 // Acceleration [x, y]
}

// Game holds the simulation state
type Game struct {
	particles []Particle
	grid      map[int][]int   // Grid maps cell index to particle indices
	pixel     *ebiten.Image   // 1x1 white pixel image for batch drawing
	vertices  []ebiten.Vertex // Pre-allocated vertex buffer
	indices   []uint16        // Pre-allocated index buffer
}

// initializeParticles creates particles with random positions
func initializeParticles(n int) []Particle {
	rand.Seed(time.Now().UnixNano())
	particles := make([]Particle, n)
	for i := range particles {
		x := rand.Float64() * Width
		y := rand.Float64() * Height
		particles[i] = Particle{
			Pos:     [2]float64{x, y},
			PrevPos: [2]float64{x, y},
			Accel:   [2]float64{0, 0},
		}
	}
	return particles
}

// buildGrid constructs a spatial grid for efficient collision detection
func (g *Game) buildGrid() {
	g.grid = make(map[int][]int)
	for i := range g.particles {
		p := &g.particles[i]
		cellX := int(p.Pos[0] / GridSize)
		cellY := int(p.Pos[1] / GridSize)
		cellIdx := cellX + cellY*(Width/GridSize)
		g.grid[cellIdx] = append(g.grid[cellIdx], i)
	}
}

// getNeighboringCells returns indices of neighboring cells (including self)
func getNeighboringCells(cellX, cellY int) []int {
	var cells []int
	for dx := -1; dx <= 1; dx++ {
		for dy := -1; dy <= 1; dy++ {
			nx := cellX + dx
			ny := cellY + dy
			if nx >= 0 && nx < Width/GridSize && ny >= 0 && ny < Height/GridSize {
				cells = append(cells, nx+ny*(Width/GridSize))
			}
		}
	}
	return cells
}

// calculateAccelerations computes repulsive forces for a chunk of particles
func (g *Game) calculateAccelerations(start, end int) {
	for i := start; i < end; i++ {
		p := &g.particles[i]
		p.Accel = [2]float64{0, 0} // Reset acceleration
		cellX := int(p.Pos[0] / GridSize)
		cellY := int(p.Pos[1] / GridSize)
		neighbors := getNeighboringCells(cellX, cellY)

		for _, cellIdx := range neighbors {
			if indices, ok := g.grid[cellIdx]; ok {
				for _, j := range indices {
					if i == j {
						continue
					}
					p2 := &g.particles[j]
					dx := p.Pos[0] - p2.Pos[0]
					dy := p.Pos[1] - p2.Pos[1]
					dist := math.Sqrt(dx*dx + dy*dy)
					if dist < MinDistance && dist > 0 {
						force := K * (MinDistance - dist) / dist
						p.Accel[0] += force * dx
						p.Accel[1] += force * dy
					}
				}
			}
		}
	}
}

// updatePositions updates particle positions for a chunk using Verlet integration
func (g *Game) updatePositions(start, end int) {
	for i := start; i < end; i++ {
		p := &g.particles[i]
		// Verlet integration: new_pos = 2*pos - prev_pos + accel*dt^2
		newX := 2*p.Pos[0] - p.PrevPos[0] + p.Accel[0]*Delta*Delta
		newY := 2*p.Pos[1] - p.PrevPos[1] + p.Accel[1]*Delta*Delta
		p.PrevPos = p.Pos
		p.Pos = [2]float64{newX, newY}

		// Wrap around screen edges
		if p.Pos[0] < 0 {
			p.Pos[0] += Width
			p.PrevPos[0] += Width
		} else if p.Pos[0] >= Width {
			p.Pos[0] -= Width
			p.PrevPos[0] -= Width
		}
		if p.Pos[1] < 0 {
			p.Pos[1] += Height
			p.PrevPos[1] += Height
		} else if p.Pos[1] >= Height {
			p.Pos[1] -= Height
			p.PrevPos[1] -= Height
		}
	}
}

// handleCollisions resolves collisions by separating overlapping particles
func (g *Game) handleCollisions() {
	for i := range g.particles {
		p := &g.particles[i]
		cellX := int(p.Pos[0] / GridSize)
		cellY := int(p.Pos[1] / GridSize)
		neighbors := getNeighboringCells(cellX, cellY)

		for _, cellIdx := range neighbors {
			if indices, ok := g.grid[cellIdx]; ok {
				for _, j := range indices {
					if i <= j { // Avoid double-checking pairs
						continue
					}
					p2 := &g.particles[j]
					dx := p.Pos[0] - p2.Pos[0]
					dy := p.Pos[1] - p2.Pos[1]
					dist := math.Sqrt(dx*dx + dy*dy)
					if dist < MinDistance && dist > 0 {
						// Separate particles
						overlap := MinDistance - dist
						moveX := dx / dist * overlap * 0.5
						moveY := dy / dist * overlap * 0.5
						p.Pos[0] += moveX
						p.Pos[1] += moveY
						p2.Pos[0] -= moveX
						p2.Pos[1] -= moveY
					}
				}
			}
		}
	}
}

// simulationStep performs one full simulation step
func (g *Game) simulationStep() {
	g.buildGrid()

	// Parallelize acceleration calculation
	var wg sync.WaitGroup
	numThreads := runtime.GOMAXPROCS(0)
	chunkSize := len(g.particles) / numThreads
	for t := 0; t < numThreads; t++ {
		start := t * chunkSize
		end := start + chunkSize
		if t == numThreads-1 {
			end = len(g.particles)
		}
		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			g.calculateAccelerations(s, e)
		}(start, end)
	}
	wg.Wait()

	// Parallelize position updates
	for t := 0; t < numThreads; t++ {
		start := t * chunkSize
		end := start + chunkSize
		if t == numThreads-1 {
			end = len(g.particles)
		}
		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			g.updatePositions(s, e)
		}(start, end)
	}
	wg.Wait()

	// Handle collisions (not parallelized for simplicity)
	g.handleCollisions()
}

// Update is called each frame by Ebitengine
func (g *Game) Update() error {
	g.simulationStep()
	return nil
}

// Draw renders visible particles using batch drawing with DrawTriangles
func (g *Game) Draw(screen *ebiten.Image) {
	// Reset vertex buffer
	vertexCount := 0

	// Half size for positioning
	halfSize := float32(ParticleSize / 2)

	// Fill vertex buffer for visible particles
	for _, p := range g.particles {
		if p.Pos[0] >= 0 && p.Pos[0] < Width && p.Pos[1] >= 0 && p.Pos[1] < Height {
			x, y := float32(p.Pos[0]), float32(p.Pos[1])

			// Define the four vertices of a square (quad)
			g.vertices[vertexCount+0] = ebiten.Vertex{
				DstX:   x - halfSize,
				DstY:   y - halfSize,
				SrcX:   0,
				SrcY:   0,
				ColorR: 1.0, // Red
				ColorG: 0.0,
				ColorB: 0.0,
				ColorA: 1.0,
			}
			g.vertices[vertexCount+1] = ebiten.Vertex{
				DstX:   x + halfSize,
				DstY:   y - halfSize,
				SrcX:   1,
				SrcY:   0,
				ColorR: 1.0,
				ColorG: 0.0,
				ColorB: 0.0,
				ColorA: 1.0,
			}
			g.vertices[vertexCount+2] = ebiten.Vertex{
				DstX:   x - halfSize,
				DstY:   y + halfSize,
				SrcX:   0,
				SrcY:   1,
				ColorR: 1.0,
				ColorG: 0.0,
				ColorB: 0.0,
				ColorA: 1.0,
			}
			g.vertices[vertexCount+3] = ebiten.Vertex{
				DstX:   x + halfSize,
				DstY:   y + halfSize,
				SrcX:   1,
				SrcY:   1,
				ColorR: 1.0,
				ColorG: 0.0,
				ColorB: 0.0,
				ColorA: 1.0,
			}

			vertexCount += 4
		}
	}

	if vertexCount > 0 {
		// Draw all visible particles in one call
		screen.DrawTriangles(g.vertices[:vertexCount], g.indices[:vertexCount/2*3], g.pixel, nil)
	}

	ebitenutil.DebugPrint(screen, fmt.Sprintf("TPS: %f\nFPS: %f\nCount: %d", ebiten.ActualTPS(), ebiten.ActualFPS(), len(g.particles)))
}

// Layout sets the screen size
func (g *Game) Layout(outsideWidth, outsideHeight int) (int, int) {
	return Width, Height
}

func main() {
	// Create a 1x1 white pixel image
	pixel := ebiten.NewImage(1, 1)
	pixel.Fill(color.White)

	// Pre-allocate vertex and index buffers (max size for all particles)
	vertices := make([]ebiten.Vertex, NumParticles*4) // 4 vertices per particle
	indices := make([]uint16, NumParticles*6)         // 6 indices per particle (2 triangles)
	for i := 0; i < NumParticles; i++ {
		base := uint16(i * 4)
		// First triangle: 0-1-2
		indices[i*6+0] = base + 0
		indices[i*6+1] = base + 1
		indices[i*6+2] = base + 2
		// Second triangle: 1-2-3
		indices[i*6+3] = base + 1
		indices[i*6+4] = base + 2
		indices[i*6+5] = base + 3
	}

	// Initialize game state
	game := &Game{
		particles: initializeParticles(NumParticles),
		pixel:     pixel,
		vertices:  vertices,
		indices:   indices,
	}

	// Set up Ebitengine
	ebiten.SetWindowSize(Width, Height)
	ebiten.SetWindowTitle("Verlet Physics Simulation")
	ebiten.SetWindowResizingMode(ebiten.WindowResizingModeEnabled)

	// Run the game
	if err := ebiten.RunGame(game); err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Simulating %d particles with %d threads\n", NumParticles, runtime.GOMAXPROCS(0))
}
