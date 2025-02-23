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
	"github.com/hajimehoshi/ebiten/v2/inpututil"
)

// Constants
const (
	screenWidth  = 800
	screenHeight = 600
	cellSize     = 8
	gridWidth    = screenWidth / cellSize
	gridHeight   = screenHeight / cellSize
	radius       = 2.0
	diameter     = radius * 2
	maxParticles = 10000
	gravity      = 0.3
	subSteps     = 8

	vertexBufferSize  = 16384
	maxBatchParticles = vertexBufferSize / 4

	constraintIterations = 1

	// Rainbow constants
	particlesPerLerp = 100
	hueCycle         = 360.0

	// Settling and performance constants
	velocityThresholdSq = 0.00001      // Tight threshold for stopping
	settlingZone        = diameter * 2 // Settle within 2 diameters of bottom
	horizontalDamping   = 0.95         // Horizontal damping near bottom
)

// Neighbor offsets for grid cells (9 surrounding cells)
var neighborOffsets = [][2]int{
	{-1, -1}, {0, -1}, {1, -1},
	{-1, 0}, {0, 0}, {1, 0},
	{-1, 1}, {0, 1}, {1, 1},
}

// WorldConfig holds static simulation boundaries
type WorldConfig struct {
	leftBound, rightBound float64
	topBound, bottomBound float64
}

// Vec2 represents a 2D vector
type Vec2 struct {
	x, y float64
}

// Particle represents a single particle with position and color
type Particle struct {
	pos, oldPos Vec2
	color       color.RGBA
}

// Grid for spatial partitioning
type Grid struct {
	cells [][]*Particle
}

// World manages the simulation state
type World struct {
	particles  []*Particle
	count      int
	grid       Grid
	config     WorldConfig
	workerPool chan func()
	redCells   [][2]int
	blackCells [][2]int
	numWorkers int
}

// newWorld initializes the simulation world
func newWorld() *World {
	w := &World{
		particles: make([]*Particle, maxParticles),
		grid: Grid{
			cells: make([][]*Particle, gridWidth*gridHeight),
		},
		config: WorldConfig{
			leftBound:   radius,
			rightBound:  screenWidth - radius,
			topBound:    radius,
			bottomBound: screenHeight - radius,
		},
		numWorkers: runtime.NumCPU(),
	}
	w.workerPool = make(chan func(), w.numWorkers)
	for i := range w.grid.cells {
		w.grid.cells[i] = make([]*Particle, 0, 32) // Pre-allocate capacity
	}
	// Create red-black checkerboard pattern for parallel processing
	for y := 0; y < gridHeight; y++ {
		for x := 0; x < gridWidth; x++ {
			if (x+y)%2 == 0 {
				w.redCells = append(w.redCells, [2]int{x, y})
			} else {
				w.blackCells = append(w.blackCells, [2]int{x, y})
			}
		}
	}
	// Start worker goroutines
	for i := 0; i < w.numWorkers; i++ {
		go func() {
			for f := range w.workerPool {
				f()
			}
		}()
	}
	return w
}

// addParticle creates a new particle with initial conditions and color
func (w *World) addParticle(x, y float64) {
	if w.count >= maxParticles {
		return
	}
	// Rainbow color calculation
	group := float64(w.count / particlesPerLerp)
	progress := float64(w.count%particlesPerLerp) / float64(particlesPerLerp)
	startHue := math.Mod(group*hueCycle/float64(maxParticles/particlesPerLerp), hueCycle)
	endHue := math.Mod((group+1)*hueCycle/float64(maxParticles/particlesPerLerp), hueCycle)
	hue := startHue + (endHue-startHue)*progress
	r, g, b := hsvToRGB(hue, 1.0, 1.0)
	particleColor := color.RGBA{R: uint8(r), G: uint8(g), B: uint8(b), A: 255}
	w.particles[w.count] = &Particle{
		pos:    Vec2{x, y},
		oldPos: Vec2{x - 1, y}, // Slight initial velocity
		color:  particleColor,
	}
	w.count++
}

// hsvToRGB converts HSV to RGB
func hsvToRGB(h, s, v float64) (uint8, uint8, uint8) {
	h = math.Mod(h, 360)
	c := v * s
	x := c * (1 - math.Abs(math.Mod(h/60, 2)-1))
	m := v - c
	var r, g, b float64
	switch {
	case 0 <= h && h < 60:
		r, g, b = c, x, 0
	case 60 <= h && h < 120:
		r, g, b = x, c, 0
	case 120 <= h && h < 180:
		r, g, b = 0, c, x
	case 180 <= h && h < 240:
		r, g, b = 0, x, c
	case 240 <= h && h < 300:
		r, g, b = x, 0, c
	case 300 <= h && h < 360:
		r, g, b = c, 0, x
	}
	return uint8(255 * (r + m)), uint8(255 * (g + m)), uint8(255 * (b + m))
}

// updateGrid assigns particles to grid cells
func (w *World) updateGrid() {
	for i := range w.grid.cells {
		w.grid.cells[i] = w.grid.cells[i][:0] // Clear without reallocating
	}
	for i := 0; i < w.count; i++ {
		p := w.particles[i]
		cellX := int(p.pos.x / cellSize)
		cellY := int(p.pos.y / cellSize)
		if cellX >= 0 && cellX < gridWidth && cellY >= 0 && cellY < gridHeight {
			idx := cellY*gridWidth + cellX
			w.grid.cells[idx] = append(w.grid.cells[idx], p)
		}
	}
}

// processCells handles constraint solving for a list of cells in parallel
func (w *World) processCells(cells [][2]int) {
	chunkSize := (len(cells) + w.numWorkers - 1) / w.numWorkers
	var wg sync.WaitGroup
	for i := 0; i < w.numWorkers; i++ {
		start := i * chunkSize
		end := min(start+chunkSize, len(cells))
		if start < end {
			wg.Add(1)
			w.workerPool <- func(start, end int) func() {
				return func() {
					for j := start; j < end; j++ {
						x, y := cells[j][0], cells[j][1]
						w.solveConstraintsCell(x, y)
					}
					wg.Done()
				}
			}(start, end)
		}
	}
	wg.Wait()
}

// solveConstraintsCell resolves collisions within a cell and its neighbors
func (w *World) solveConstraintsCell(cellX, cellY int) {
	idx := cellY*gridWidth + cellX
	particles := w.grid.cells[idx]
	for i, p1 := range particles {
		// Check within the same cell
		for j := i + 1; j < len(particles); j++ {
			p2 := particles[j]
			w.solveParticlePair(p1, p2)
		}
		// Check neighbor cells
		for _, offset := range neighborOffsets {
			nCellX := cellX + offset[0]
			nCellY := cellY + offset[1]
			if nCellX >= 0 && nCellX < gridWidth && nCellY >= 0 && nCellY < gridHeight {
				neighborIdx := nCellY*gridWidth + nCellX
				for _, p2 := range w.grid.cells[neighborIdx] {
					if p1 != p2 {
						w.solveParticlePair(p1, p2)
					}
				}
			}
		}
	}
}

// solveParticlePair resolves collision between two particles with settling optimization
func (w *World) solveParticlePair(p1, p2 *Particle) {
	// Check if both particles are near bottom and nearly still
	v1x := p1.pos.x - p1.oldPos.x
	v1y := p1.pos.y - p1.oldPos.y
	v2x := p2.pos.x - p2.oldPos.x
	v2y := p2.pos.y - p2.oldPos.y
	vel1Sq := v1x*v1x + v1y*v1y
	vel2Sq := v2x*v2x + v2y*v2y
	if p1.pos.y > w.config.bottomBound-settlingZone && p2.pos.y > w.config.bottomBound-settlingZone &&
		vel1Sq < velocityThresholdSq && vel2Sq < velocityThresholdSq {
		return // Skip collision if both are settled at bottom
	}

	// Proceed with collision resolution
	dx := p1.pos.x - p2.pos.x
	dy := p1.pos.y - p2.pos.y
	distSq := dx*dx + dy*dy
	if distSq < diameter*diameter && distSq > 0.00001 { // Check overlap
		dist := math.Sqrt(distSq)
		delta := (diameter - dist) / (dist * 2)
		moveX := dx * delta
		moveY := dy * delta
		p1.pos.x += moveX
		p1.pos.y += moveY
		p2.pos.x -= moveX
		p2.pos.y -= moveY
	}
}

// update advances the simulation with substeps
func (w *World) update() {
	dt := 1.0 / float64(subSteps)
	for step := 0; step < subSteps; step++ {
		// Update positions with Verlet integration
		for i := 0; i < w.count; i++ {
			p := w.particles[i]
			dx := p.pos.x - p.oldPos.x
			dy := p.pos.y - p.oldPos.y
			// Apply horizontal damping only near bottom
			if p.pos.y > w.config.bottomBound-settlingZone {
				dx *= horizontalDamping
			}
			temp := p.pos
			p.pos.x += dx
			p.pos.y += dy + gravity*dt*dt
			p.oldPos = temp
		}
		w.updateGrid()
		// Solve constraints in red-black order
		for iter := 0; iter < constraintIterations; iter++ {
			w.processCells(w.redCells)
			w.processCells(w.blackCells)
		}
		// Apply boundaries and stop small movements near bottom
		for i := 0; i < w.count; i++ {
			p := w.particles[i]
			dx := p.pos.x - p.oldPos.x
			dy := p.pos.y - p.oldPos.y
			velSq := dx*dx + dy*dy
			// Stop particles near bottom with low velocity
			if p.pos.y > w.config.bottomBound-settlingZone && velSq < velocityThresholdSq {
				p.pos = p.oldPos // Reset to stop movement
			}
			// Enforce boundaries
			if p.pos.x < w.config.leftBound {
				p.pos.x = w.config.leftBound
				p.oldPos.x = p.pos.x - dx*0.3
			} else if p.pos.x > w.config.rightBound {
				p.pos.x = w.config.rightBound
				p.oldPos.x = p.pos.x - dx*0.3
			}
			if p.pos.y < w.config.topBound {
				p.pos.y = w.config.topBound
				p.oldPos.y = p.pos.y - dy*0.3
			} else if p.pos.y > w.config.bottomBound {
				p.pos.y = w.config.bottomBound
				p.oldPos.y = p.pos.y
			}
		}
	}
}

// reset clears all particles
func (w *World) reset() {
	w.count = 0
	for i := range w.grid.cells {
		w.grid.cells[i] = w.grid.cells[i][:0]
	}
}

// Game implements the Ebitengine game interface
type Game struct {
	world    *World
	pixels   *ebiten.Image
	vertices []ebiten.Vertex
	indices  []uint16
}

// Update handles input and simulation updates
func (g *Game) Update() error {
	if inpututil.IsKeyJustPressed(ebiten.KeySpace) {
		g.world.reset()
		return nil
	}
	if g.world.count < maxParticles {
		for i := 0; i < 50; i++ {
			g.world.addParticle(20+rand.Float64()*10, 20+rand.Float64()*10)
		}
	}
	g.world.update()
	return nil
}

// Draw renders particles with their respective colors and debug info
func (g *Game) Draw(screen *ebiten.Image) {
	const particleSize = float32(radius * 2)
	particlesPerBatch := maxBatchParticles
	for batch := 0; batch < (g.world.count+particlesPerBatch-1)/particlesPerBatch; batch++ {
		start := batch * particlesPerBatch
		end := min(start+particlesPerBatch, g.world.count)
		vertexCount := 0
		for i := start; i < end; i++ {
			p := g.world.particles[i]
			x, y := float32(p.pos.x), float32(p.pos.y)
			c := p.color
			base := vertexCount
			g.vertices[base] = ebiten.Vertex{
				DstX:   x - particleSize/2,
				DstY:   y - particleSize/2,
				SrcX:   0,
				SrcY:   0,
				ColorR: float32(c.R) / 255,
				ColorG: float32(c.G) / 255,
				ColorB: float32(c.B) / 255,
				ColorA: 1,
			}
			g.vertices[base+1] = ebiten.Vertex{
				DstX:   x + particleSize/2,
				DstY:   y - particleSize/2,
				SrcX:   1,
				SrcY:   0,
				ColorR: float32(c.R) / 255,
				ColorG: float32(c.G) / 255,
				ColorB: float32(c.B) / 255,
				ColorA: 1,
			}
			g.vertices[base+2] = ebiten.Vertex{
				DstX:   x - particleSize/2,
				DstY:   y + particleSize/2,
				SrcX:   0,
				SrcY:   1,
				ColorR: float32(c.R) / 255,
				ColorG: float32(c.G) / 255,
				ColorB: float32(c.B) / 255,
				ColorA: 1,
			}
			g.vertices[base+3] = ebiten.Vertex{
				DstX:   x + particleSize/2,
				DstY:   y + particleSize/2,
				SrcX:   1,
				SrcY:   1,
				ColorR: float32(c.R) / 255,
				ColorG: float32(c.G) / 255,
				ColorB: float32(c.B) / 255,
				ColorA: 1,
			}
			vertexCount += 4
		}
		if vertexCount > 0 {
			screen.DrawTriangles(g.vertices[:vertexCount], g.indices[:vertexCount/2*3], g.pixels, &ebiten.DrawTrianglesOptions{})
		}
	}
	ebitenutil.DebugPrint(screen, fmt.Sprintf("TPS: %f\nFPS: %.0f\nParticles: %d\nPress [space] to reset", ebiten.ActualTPS(), ebiten.ActualFPS(), g.world.count))
}

// Layout sets the screen size
func (g *Game) Layout(w, h int) (int, int) {
	return screenWidth, screenHeight
}

// main sets up and runs the game
func main() {
	rand.Seed(time.Now().UnixNano())
	game := &Game{
		world:    newWorld(),
		pixels:   ebiten.NewImage(1, 1),
		vertices: make([]ebiten.Vertex, vertexBufferSize),
		indices:  make([]uint16, vertexBufferSize/2*3),
	}
	game.pixels.Fill(color.White)
	// Precompute triangle indices for rendering
	for i := 0; i < maxBatchParticles; i++ {
		base := uint16(i * 4)
		idx := i * 6
		game.indices[idx+0] = base + 0
		game.indices[idx+1] = base + 1
		game.indices[idx+2] = base + 2
		game.indices[idx+3] = base + 1
		game.indices[idx+4] = base + 2
		game.indices[idx+5] = base + 3
	}
	ebiten.SetWindowSize(screenWidth, screenHeight)
	ebiten.SetWindowTitle("Verlet Multithread Simulation")
	if err := ebiten.RunGame(game); err != nil {
		log.Fatal(err)
	}
}

// min helper function
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
