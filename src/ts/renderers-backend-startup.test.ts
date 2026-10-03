import { afterAll, afterEach, beforeAll, describe, expect, spyOn, test } from 'bun:test'
import { PgsRenderer } from './renderers'
import { WebGPURenderer } from './webgpu-renderer'
import { WebGL2Renderer } from './webgl2-renderer'
import { initWasm } from './wasm'
import type { SubtitleData } from './types'

const saved = Object.fromEntries(
  ['window', 'document', 'ResizeObserver', 'requestAnimationFrame', 'cancelAnimationFrame'].map((key) => [
    key,
    (globalThis as any)[key]
  ])
)
const draws: number[] = []
let canvasDraws = 0
let clears = 0
const context = {
  clearRect() {
    clears++
  },
  save() {},
  restore() {},
  putImageData() {},
  drawImage() {
    canvasDraws++
  },
  globalAlpha: 1
}
const data = (index: number): SubtitleData =>
  ({
    width: 2,
    height: 2,
    compositionData: [
      { x: index, y: 0, pixelData: { width: 1, height: 1, data: new Uint8ClampedArray([255, 255, 255, 255]) } }
    ]
  }) as SubtitleData

// Use real canvas/backend selection and presentation paths, with deterministic decoded cues.
let subtitleGate: Promise<void> | null = null

class StartupRenderer extends PgsRenderer {
  protected async init(): Promise<void> {
    await Promise.resolve()
    this.createCanvas()
    if (subtitleGate) await subtitleGate
    this.isLoaded = true
    this.startRenderLoop()
  }
  protected findCurrentIndex(time: number): number {
    return time < 2 ? Math.floor(time) : -1
  }
  protected renderAtIndex(index: number): SubtitleData | null {
    return data(index)
  }
  tick(): void {
    ;(this as any).renderSynchronizedFrame({ mediaTime: this.video.currentTime, presentedFrames: null })
  }
}
function deferred() {
  let resolve!: () => void
  let reject!: (reason: Error) => void
  const promise = new Promise<void>((yes, no) => {
    resolve = yes
    reject = no
  })
  return { promise, resolve, reject }
}
async function settle() {
  // Flush backend continuations started through the fire-and-forget selection path.
  await new Promise<void>((resolve) => setTimeout(resolve, 0))
}
function video(paused = true): HTMLVideoElement {
  return {
    currentTime: 0,
    paused,
    ended: false,
    videoWidth: 2,
    videoHeight: 2,
    addEventListener() {},
    removeEventListener() {},
    getBoundingClientRect: () => ({ x: 0, y: 0, width: 2, height: 2 })
  } as unknown as HTMLVideoElement
}
const active: PgsRenderer[] = []
function create(backend: 'webgpu' | 'webgl2' | 'canvas2d', paused = true) {
  const v = video(paused)
  const renderer = new StartupRenderer({ video: v, backend, offscreenRender: false })
  active.push(renderer)
  return { renderer, video: v }
}
beforeAll(() => {
  Object.assign(globalThis, {
    window: { devicePixelRatio: 1, getComputedStyle: () => ({ position: 'static' }) },
    document: {
      createElement: () => ({
        width: 2,
        height: 2,
        style: {},
        getContext: (kind: string) => (kind === '2d' ? context : kind === 'webgl2' ? {} : null)
      })
    },
    ResizeObserver: class {
      observe() {}
      disconnect() {}
    },
    requestAnimationFrame: () => 1,
    cancelAnimationFrame() {}
  })
})
afterEach(() => {
  active.splice(0).forEach((r) => r.dispose())
  mockRestore()
  draws.length = 0
  canvasDraws = 0
  clears = 0
  subtitleGate = null
})
const mocks: Array<{ mockRestore(): void }> = []
function mockRestore() {
  mocks.splice(0).forEach((m) => m.mockRestore())
}
function backendMocks(kind: 'webgpu' | 'webgl2', gate: ReturnType<typeof deferred>, stage = 'init') {
  const proto = kind === 'webgpu' ? WebGPURenderer.prototype : WebGL2Renderer.prototype
  const live = new Set<object>()
  function allocate(this: object): Promise<void> {
    return gate.promise.then(
      () => {
        live.add(this)
      },
      (error) => {
        live.add(this)
        throw error
      }
    )
  }
  mocks.push(
    spyOn(proto, 'init').mockImplementation(function () {
      return stage === 'init' ? allocate.call(this) : Promise.resolve()
    })
  )
  const canvas = spyOn(proto, 'setCanvas').mockImplementation(function () {
    return stage === 'setCanvas' ? allocate.call(this) : Promise.resolve()
  })
  mocks.push(
    canvas,
    spyOn(proto, 'render').mockImplementation((compositions) => {
      draws.push(compositions[0].x)
    })
  )
  const clear = spyOn(proto, 'clear').mockImplementation(() => {
    clears++
  })
  const destroy = spyOn(proto, 'destroy').mockImplementation(function () {
    live.delete(this)
  })
  mocks.push(clear, destroy)
  return { canvas, destroy, clear, live }
}
afterAll(() => Object.assign(globalThis, saved))
describe('graphics backend startup', () => {
  test('automatic WebGPU failure starts WebGL2 and redraws only when it becomes ready', async () => {
    const gpu = deferred()
    const gl = deferred()
    backendMocks('webgpu', gpu)
    backendMocks('webgl2', gl)
    const { renderer } = create('canvas2d')
    await settle()
    const startup = (renderer as any).initWebGPU()
    gpu.reject(new Error('no GPU'))
    await startup
    expect(draws).toEqual([])
    gl.resolve()
    await settle()
    expect(draws).toEqual([0])
    expect((renderer as any).currentRendererBackend).toBe('webgl2')
  })

  for (const kind of ['webgpu', 'webgl2'] as const) {
    for (const failure of ['draw', 'event'])
      test(`${kind} preserves healthy resources when ${failure} throws`, async () => {
        const gate = deferred()
        gate.resolve()
        const { destroy, live } = backendMocks(kind, gate)
        const { renderer } = create('canvas2d')
        await settle()
        const error = new Error('consumer presentation error')
        if (failure === 'draw')
          (renderer as any).renderFrame = () => {
            throw error
          }
        else
          (renderer as any).onEvent = () => {
            throw error
          }
        const fallback = spyOn(renderer as any, kind === 'webgpu' ? 'preferWorkerOffscreenOrCanvas2D' : 'initCanvas2D')
        mocks.push(fallback)
        await expect((renderer as any)[kind === 'webgpu' ? 'initWebGPU' : 'initWebGL2']()).rejects.toThrow(error)
        expect(destroy).not.toHaveBeenCalled()
        expect(fallback).not.toHaveBeenCalled()
        expect(live.size).toBe(1)
      })

    test(`${kind} readiness replaces the pending decode presentation observer`, async () => {
      const gate = deferred()
      backendMocks(kind, gate)
      const decode = deferred()
      let decoded = false
      const pending = decode.promise.then(() => {
        decoded = true
        return data(0)
      })
      class PendingRenderer extends StartupRenderer {
        protected renderAtIndex(index: number): any {
          if (decoded) return data(index)
          this.watchPendingRender(index, pending)
          return undefined
        }
        protected isPendingRender(): boolean {
          return !decoded
        }
      }
      const renderer = new PendingRenderer({ video: video(), backend: kind, offscreenRender: false })
      active.push(renderer)
      await (renderer as any).initPromise
      const token = (renderer as any).getPresentationToken()
      gate.resolve()
      await settle()
      expect((renderer as any).getPresentationToken()).toBeGreaterThan(token)
      expect(draws).toEqual([])
      decode.resolve()
      await pending
      await settle()
      expect(draws).toEqual([0])
    })
  }

  for (const stage of ['before-canvas', 'before-loading', 'loading', 'offscreen'])
    test(`real base init stops after disposal during ${stage}`, async () => {
      await initWasm()
      const entered = deferred()
      const released = deferred()
      class LifecycleRenderer extends PgsRenderer {
        protected createCanvas(): void {
          super.createCanvas()
          if (stage === 'before-loading') entered.resolve()
        }
        protected async loadSubtitles(): Promise<void> {
          if (stage === 'loading') {
            entered.resolve()
            await released.promise
          }
          this.isLoaded = true
        }
        protected async ensureWorkerOffscreenAttached(): Promise<void> {
          if (stage === 'offscreen') {
            entered.resolve()
            await released.promise
          }
        }
      }
      const renderer = new LifecycleRenderer({ video: video(), backend: 'canvas2d' })
      active.push(renderer)
      if (stage !== 'before-canvas') await entered.promise
      renderer.dispose()
      released.resolve()
      await (renderer as any).initPromise
      expect((renderer as any).canvas).toBeNull()
      expect((renderer as any).frameScheduler).toBeNull()
      expect((renderer as any).resizeObserver).toBeNull()
    })

  for (const kind of ['webgpu', 'webgl2'] as const) {
    test(`${kind} redraws the paused cue decoded before readiness`, async () => {
      const gate = deferred()
      backendMocks(kind, gate)
      const { renderer, video: v } = create(kind)
      await settle()
      expect(renderer.getStats().currentIndex).toBe(0)
      expect(draws).toEqual([])
      gate.resolve()
      await settle()
      expect(draws).toEqual([0])
      expect(v.currentTime).toBe(0)
      expect(v.paused).toBe(true)
    })
    test(`${kind} playing startup draws on the next synchronized frame`, async () => {
      const gate = deferred()
      backendMocks(kind, gate)
      const { renderer } = create(kind, false)
      await settle()
      renderer.tick()
      gate.resolve()
      await settle()
      renderer.tick()
      expect(draws).toEqual([0])
    })
    for (const time of [1, 2])
      test(`${kind} reselects at current time ${time}`, async () => {
        const gate = deferred()
        const { clear } = backendMocks(kind, gate)
        const { video: v } = create(kind)
        await settle()
        v.currentTime = time
        gate.resolve()
        await settle()
        expect(draws).toEqual(time === 1 ? [1] : [])
        if (time === 2) expect(clear).toHaveBeenCalledTimes(1)
      })
    test(`${kind} failure falls back and draws the paused cue`, async () => {
      const gate = deferred()
      backendMocks(kind, gate)
      create(kind)
      await settle()
      gate.reject(new Error('unavailable'))
      await settle()
      expect(canvasDraws).toBe(1)
    })
    for (const stage of ['init', 'setCanvas'])
      for (const fails of [false, true]) {
        test(`${kind} disposal during ${stage} ${fails ? 'failure' : 'success'} releases late work`, async () => {
          const gate = deferred()
          const { destroy, canvas, live } = backendMocks(kind, gate, stage)
          const { renderer } = create(kind)
          await settle()
          renderer.dispose()
          const destroyed = destroy.mock.calls.length
          if (fails) gate.reject(new Error('unavailable'))
          else gate.resolve()
          await settle()
          expect(destroy.mock.calls.length).toBeGreaterThan(destroyed)
          expect(live.size).toBe(0)
          expect(draws).toEqual([])
          expect(canvasDraws).toBe(0)
          if (stage === 'init') expect(canvas).not.toHaveBeenCalled()
          expect((renderer as any).canvas).toBeNull()
        })
      }
    test(`${kind} early readiness renders after subtitle initialization`, async () => {
      const gate = deferred()
      gate.resolve()
      backendMocks(kind, gate)
      const loading = deferred()
      subtitleGate = loading.promise
      const { renderer } = create(kind)
      await settle()
      expect(draws).toEqual([])
      expect((renderer as any).currentRendererBackend).toBe(kind)
      subtitleGate = null
      loading.resolve()
      await settle()
      expect(draws).toEqual([0])
    })
  }
})
