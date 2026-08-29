import { useCallback, useEffect, useRef, useState } from 'react'

const MIME_TYPES = [
  'video/mp4;codecs=h264',
  'video/webm;codecs=vp9',
  'video/webm;codecs=vp8',
  'video/webm',
]

function supportedMimeType() {
  if (typeof MediaRecorder === 'undefined') return ''
  return MIME_TYPES.find((candidate) => MediaRecorder.isTypeSupported(candidate)) ?? ''
}

function safeFilename(value: string) {
  return value.replace(/[^a-z0-9._-]+/gi, '-').replace(/^-+|-+$/g, '') || 'swarmworld-replay'
}

export function useVideoExport() {
  const recorderRef = useRef<MediaRecorder | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const filenameRef = useRef('swarmworld-replay')
  const [recording, setRecording] = useState(false)
  const [error, setError] = useState('')

  const stop = useCallback(() => {
    const recorder = recorderRef.current
    if (recorder && recorder.state !== 'inactive') recorder.stop()
    streamRef.current?.getTracks().forEach((track) => track.stop())
  }, [])

  const start = useCallback(async (filename: string) => {
    if (recording) return
    setError('')
    if (!navigator.mediaDevices?.getDisplayMedia || typeof MediaRecorder === 'undefined') {
      const message = 'This browser does not support tab video recording.'
      setError(message)
      throw new Error(message)
    }
    const stream = await navigator.mediaDevices.getDisplayMedia({
      video: { frameRate: { ideal: 30, max: 60 } },
      audio: false,
      preferCurrentTab: true,
      selfBrowserSurface: 'include',
      surfaceSwitching: 'exclude',
    } as DisplayMediaStreamOptions)
    const mimeType = supportedMimeType()
    const recorder = new MediaRecorder(stream, {
      ...(mimeType ? { mimeType } : {}),
      videoBitsPerSecond: 12_000_000,
    })
    chunksRef.current = []
    filenameRef.current = safeFilename(filename)
    recorderRef.current = recorder
    streamRef.current = stream
    recorder.ondataavailable = (event) => {
      if (event.data.size) chunksRef.current.push(event.data)
    }
    recorder.onstop = () => {
      const actualType = recorder.mimeType || mimeType || 'video/webm'
      const extension = actualType.includes('mp4') ? 'mp4' : 'webm'
      const blob = new Blob(chunksRef.current, { type: actualType })
      if (blob.size) {
        const url = URL.createObjectURL(blob)
        const link = document.createElement('a')
        link.href = url
        link.download = `${filenameRef.current}.${extension}`
        document.body.appendChild(link)
        link.click()
        link.remove()
        window.setTimeout(() => URL.revokeObjectURL(url), 2_000)
      }
      stream.getTracks().forEach((track) => track.stop())
      recorderRef.current = null
      streamRef.current = null
      chunksRef.current = []
      setRecording(false)
    }
    recorder.onerror = () => {
      setError('The browser stopped the video recorder unexpectedly.')
    }
    stream.getVideoTracks().forEach((track) => {
      track.onended = () => {
        if (recorder.state !== 'inactive') recorder.stop()
      }
    })
    recorder.start(500)
    setRecording(true)
  }, [recording])

  useEffect(() => () => {
    streamRef.current?.getTracks().forEach((track) => track.stop())
  }, [])

  return { recording, error, start, stop }
}
