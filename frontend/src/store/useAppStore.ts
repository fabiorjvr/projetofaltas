import { create } from 'zustand'
import { devtools, persist } from 'zustand/middleware'
import { immer } from 'zustand/middleware/immer'

interface Player {
  id: string
  name: string
  team: string
  position: string
  fouls: number
  cards: number
  avatar?: string
}

interface Team {
  id: string
  name: string
  logo?: string
  players: Player[]
}

interface Prediction {
  id: string
  playerId: string
  playerName: string
  matchDate: string
  foulsPredicted: number
  cardsPredicted: number
  confidence: number
  status: 'pending' | 'completed' | 'cancelled'
}

interface UIState {
  sidebarOpen: boolean
  theme: 'light' | 'dark'
  loading: boolean
  notifications: Notification[]
  selectedPlayer: Player | null
  selectedTeam: Team | null
}

interface Notification {
  id: string
  type: 'success' | 'error' | 'warning' | 'info'
  title: string
  message: string
  timestamp: number
  read: boolean
}

interface AppState {
  // Data
  players: Player[]
  teams: Team[]
  predictions: Prediction[]
  
  // UI State
  ui: UIState
  
  // Actions
  setPlayers: (players: Player[]) => void
  addPlayer: (player: Player) => void
  updatePlayer: (id: string, updates: Partial<Player>) => void
  removePlayer: (id: string) => void
  
  setTeams: (teams: Team[]) => void
  addTeam: (team: Team) => void
  updateTeam: (id: string, updates: Partial<Team>) => void
  removeTeam: (id: string) => void
  
  setPredictions: (predictions: Prediction[]) => void
  addPrediction: (prediction: Prediction) => void
  updatePrediction: (id: string, updates: Partial<Prediction>) => void
  removePrediction: (id: string) => void
  
  // UI Actions
  toggleSidebar: () => void
  setSidebarOpen: (open: boolean) => void
  setTheme: (theme: 'light' | 'dark') => void
  setLoading: (loading: boolean) => void
  setSelectedPlayer: (player: Player | null) => void
  setSelectedTeam: (team: Team | null) => void
  
  // Notifications
  addNotification: (notification: Omit<Notification, 'id' | 'timestamp' | 'read'>) => void
  markNotificationRead: (id: string) => void
  removeNotification: (id: string) => void
  clearNotifications: () => void
  
  // Bulk actions
  reset: () => void
}

const initialState = {
  players: [],
  teams: [],
  predictions: [],
  ui: {
    sidebarOpen: true,
    theme: 'light' as const,
    loading: false,
    notifications: [],
    selectedPlayer: null,
    selectedTeam: null,
  },
}

export const useAppStore = create<AppState>()()
  devtools(
    persist(
      immer((set, get) => ({
        ...initialState,
        
        // Player actions
        setPlayers: (players) => set((state) => {
          state.players = players
        }),
        
        addPlayer: (player) => set((state) => {
          state.players.push(player)
        }),
        
        updatePlayer: (id, updates) => set((state) => {
          const index = state.players.findIndex(p => p.id === id)
          if (index !== -1) {
            Object.assign(state.players[index], updates)
          }
        }),
        
        removePlayer: (id) => set((state) => {
          state.players = state.players.filter(p => p.id !== id)
          if (state.ui.selectedPlayer?.id === id) {
            state.ui.selectedPlayer = null
          }
        }),
        
        // Team actions
        setTeams: (teams) => set((state) => {
          state.teams = teams
        }),
        
        addTeam: (team) => set((state) => {
          state.teams.push(team)
        }),
        
        updateTeam: (id, updates) => set((state) => {
          const index = state.teams.findIndex(t => t.id === id)
          if (index !== -1) {
            Object.assign(state.teams[index], updates)
          }
        }),
        
        removeTeam: (id) => set((state) => {
          state.teams = state.teams.filter(t => t.id !== id)
          if (state.ui.selectedTeam?.id === id) {
            state.ui.selectedTeam = null
          }
        }),
        
        // Prediction actions
        setPredictions: (predictions) => set((state) => {
          state.predictions = predictions
        }),
        
        addPrediction: (prediction) => set((state) => {
          state.predictions.push(prediction)
        }),
        
        updatePrediction: (id, updates) => set((state) => {
          const index = state.predictions.findIndex(p => p.id === id)
          if (index !== -1) {
            Object.assign(state.predictions[index], updates)
          }
        }),
        
        removePrediction: (id) => set((state) => {
          state.predictions = state.predictions.filter(p => p.id !== id)
        }),
        
        // UI actions
        toggleSidebar: () => set((state) => {
          state.ui.sidebarOpen = !state.ui.sidebarOpen
        }),
        
        setSidebarOpen: (open) => set((state) => {
          state.ui.sidebarOpen = open
        }),
        
        setTheme: (theme) => set((state) => {
          state.ui.theme = theme
        }),
        
        setLoading: (loading) => set((state) => {
          state.ui.loading = loading
        }),
        
        setSelectedPlayer: (player) => set((state) => {
          state.ui.selectedPlayer = player
        }),
        
        setSelectedTeam: (team) => set((state) => {
          state.ui.selectedTeam = team
        }),
        
        // Notification actions
        addNotification: (notification) => set((state) => {
          const newNotification: Notification = {
            ...notification,
            id: Date.now().toString(),
            timestamp: Date.now(),
            read: false,
          }
          state.ui.notifications.unshift(newNotification)
          
          // Keep only last 50 notifications
          if (state.ui.notifications.length > 50) {
            state.ui.notifications = state.ui.notifications.slice(0, 50)
          }
        }),
        
        markNotificationRead: (id) => set((state) => {
          const notification = state.ui.notifications.find(n => n.id === id)
          if (notification) {
            notification.read = true
          }
        }),
        
        removeNotification: (id) => set((state) => {
          state.ui.notifications = state.ui.notifications.filter(n => n.id !== id)
        }),
        
        clearNotifications: () => set((state) => {
          state.ui.notifications = []
        }),
        
        // Reset
        reset: () => set(() => ({ ...initialState })),
      })),
      {
        name: 'football-fouls-app-store',
        partialize: (state) => ({
          ui: {
            theme: state.ui.theme,
            sidebarOpen: state.ui.sidebarOpen,
          },
          // Don't persist sensitive data
        }),
      }
    ),
    {
      name: 'football-fouls-store',
    }
  )

// Selectors for better performance
export const usePlayersCount = () => useAppStore(state => state.players.length)
export const useTeamsCount = () => useAppStore(state => state.teams.length)
export const usePredictionsCount = () => useAppStore(state => state.predictions.length)
export const useUnreadNotifications = () => useAppStore(state => 
  state.ui.notifications.filter(n => !n.read).length
)
export const useTheme = () => useAppStore(state => state.ui.theme)
export const useSidebarOpen = () => useAppStore(state => state.ui.sidebarOpen)
export const useLoading = () => useAppStore(state => state.ui.loading)