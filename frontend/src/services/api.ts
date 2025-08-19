const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// Types
export interface DashboardStats {
  totalPlayers: number
  totalTeams: number
  totalFouls: number
  totalPredictions: number
  playersTrend: number
  teamsTrend: number
  foulsTrend: number
  predictionsTrend: number
  foulsByMonth: Array<{ name: string; value: number }>
  cardsByLeague: Array<{ name: string; value: number }>
  recentMatches: Array<{
    id: string
    homeTeam: string
    awayTeam: string
    homeScore?: number
    awayScore?: number
    date: string
    status: 'scheduled' | 'live' | 'finished'
    venue?: string
    foulsHome?: number
    foulsAway?: number
    cardsHome?: number
    cardsAway?: number
  }>
  topPlayers: Array<{
    id: string
    name: string
    team: string
    position: string
    fouls: number
    cards: number
    matches: number
    avatar?: string
    riskLevel: 'low' | 'medium' | 'high'
  }>
}

export interface Player {
  id: string
  name: string
  team: string
  position: string
  age: number
  nationality: string
  fouls: number
  cards: number
  matches: number
  avatar?: string
  riskLevel: 'low' | 'medium' | 'high'
  stats: {
    foulsPerMatch: number
    cardsPerMatch: number
    yellowCards: number
    redCards: number
  }
}

export interface Team {
  id: string
  name: string
  league: string
  logo?: string
  stats: {
    matches: number
    fouls: number
    cards: number
    avgFoulsPerMatch: number
  }
}

export interface Match {
  id: string
  homeTeam: string
  awayTeam: string
  homeScore?: number
  awayScore?: number
  date: string
  status: 'scheduled' | 'live' | 'finished'
  venue?: string
  league: string
  stats?: {
    foulsHome: number
    foulsAway: number
    cardsHome: number
    cardsAway: number
  }
}

export interface Prediction {
  id: string
  matchId: string
  playerName: string
  team: string
  predictionType: 'foul' | 'yellow_card' | 'red_card'
  probability: number
  confidence: 'low' | 'medium' | 'high'
  createdAt: string
}

// API Client
class ApiClient {
  private baseUrl: string
  private token: string | null = null

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl
    this.token = localStorage.getItem('auth_token')
  }

  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`
    
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
      ...options.headers,
    }

    if (this.token) {
      headers.Authorization = `Bearer ${this.token}`
    }

    const response = await fetch(url, {
      ...options,
      headers,
    })

    if (!response.ok) {
      throw new Error(`API Error: ${response.status} ${response.statusText}`)
    }

    return response.json()
  }

  // Auth methods
  setToken(token: string) {
    this.token = token
    localStorage.setItem('auth_token', token)
  }

  clearToken() {
    this.token = null
    localStorage.removeItem('auth_token')
  }

  // Dashboard
  async getDashboardStats(): Promise<DashboardStats> {
    return this.request<DashboardStats>('/api/dashboard/stats')
  }

  // Players
  async getPlayers(params?: {
    page?: number
    limit?: number
    search?: string
    team?: string
    position?: string
  }): Promise<{ players: Player[]; total: number; page: number; totalPages: number }> {
    const searchParams = new URLSearchParams()
    if (params?.page) searchParams.set('page', params.page.toString())
    if (params?.limit) searchParams.set('limit', params.limit.toString())
    if (params?.search) searchParams.set('search', params.search)
    if (params?.team) searchParams.set('team', params.team)
    if (params?.position) searchParams.set('position', params.position)

    const query = searchParams.toString()
    return this.request<{ players: Player[]; total: number; page: number; totalPages: number }>(
      `/api/players${query ? `?${query}` : ''}`
    )
  }

  async getPlayer(id: string): Promise<Player> {
    return this.request<Player>(`/api/players/${id}`)
  }

  // Teams
  async getTeams(): Promise<Team[]> {
    return this.request<Team[]>('/api/teams')
  }

  async getTeam(id: string): Promise<Team> {
    return this.request<Team>(`/api/teams/${id}`)
  }

  // Matches
  async getMatches(params?: {
    page?: number
    limit?: number
    status?: string
    team?: string
  }): Promise<{ matches: Match[]; total: number; page: number; totalPages: number }> {
    const searchParams = new URLSearchParams()
    if (params?.page) searchParams.set('page', params.page.toString())
    if (params?.limit) searchParams.set('limit', params.limit.toString())
    if (params?.status) searchParams.set('status', params.status)
    if (params?.team) searchParams.set('team', params.team)

    const query = searchParams.toString()
    return this.request<{ matches: Match[]; total: number; page: number; totalPages: number }>(
      `/api/matches${query ? `?${query}` : ''}`
    )
  }

  async getMatch(id: string): Promise<Match> {
    return this.request<Match>(`/api/matches/${id}`)
  }

  // Predictions
  async getPredictions(params?: {
    page?: number
    limit?: number
    matchId?: string
    playerId?: string
  }): Promise<{ predictions: Prediction[]; total: number; page: number; totalPages: number }> {
    const searchParams = new URLSearchParams()
    if (params?.page) searchParams.set('page', params.page.toString())
    if (params?.limit) searchParams.set('limit', params.limit.toString())
    if (params?.matchId) searchParams.set('match_id', params.matchId)
    if (params?.playerId) searchParams.set('player_id', params.playerId)

    const query = searchParams.toString()
    return this.request<{ predictions: Prediction[]; total: number; page: number; totalPages: number }>(
      `/api/predictions${query ? `?${query}` : ''}`
    )
  }

  async createPrediction(data: {
    matchId: string
    playerId: string
    predictionType: string
  }): Promise<Prediction> {
    return this.request<Prediction>('/api/predictions', {
      method: 'POST',
      body: JSON.stringify(data),
    })
  }
}

// Create API client instance
const apiClient = new ApiClient(API_BASE_URL)

// Export convenience functions
export const fetchDashboardStats = () => apiClient.getDashboardStats()
export const fetchPlayers = (params?: Parameters<typeof apiClient.getPlayers>[0]) => 
  apiClient.getPlayers(params)
export const fetchPlayer = (id: string) => apiClient.getPlayer(id)
export const fetchTeams = () => apiClient.getTeams()
export const fetchTeam = (id: string) => apiClient.getTeam(id)
export const fetchMatches = (params?: Parameters<typeof apiClient.getMatches>[0]) => 
  apiClient.getMatches(params)
export const fetchMatch = (id: string) => apiClient.getMatch(id)
export const fetchPredictions = (params?: Parameters<typeof apiClient.getPredictions>[0]) => 
  apiClient.getPredictions(params)
export const createPrediction = (data: Parameters<typeof apiClient.createPrediction>[0]) => 
  apiClient.createPrediction(data)

export default apiClient