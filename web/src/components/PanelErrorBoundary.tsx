import { Component, type ErrorInfo, type ReactNode } from 'react'

interface Props {
  children: ReactNode
  title: string
}

interface State {
  error: Error | null
}

export class PanelErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error(`${this.props.title} failed`, error, info.componentStack)
  }

  private reset = () => this.setState({ error: null })

  render() {
    if (!this.state.error) return this.props.children
    return (
      <div className="network-view network-error" role="alert">
        <strong>{this.props.title} is temporarily unavailable.</strong>
        <span>The world view remains active and retains the latest authoritative state.</span>
        <button type="button" onClick={this.reset}>Retry this panel</button>
      </div>
    )
  }
}
