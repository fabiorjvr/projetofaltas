import React, { useState, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ReactQueryDevtools } from '@tanstack/react-query-devtools'
import { Toaster } from 'react-hot-toast'
import { ThemeProvider } from './contexts/ThemeContext'
import { AuthProvider } from './contexts/AuthContext'
import Layout from './components/Layout/Layout'
import Dashboard from './pages/Dashboard'
import Players from './pages/Players'
import Teams from './pages/Teams'
import Predictions from './pages/Predictions'
import Analytics from './pages/Analytics'
import Login from './pages/Login'
import Register from './pages/Register'
import ProtectedRoute from './components/Auth/ProtectedRoute'
import { PWAInstallPrompt, PWAUpdatePrompt, OfflineIndicator } from './components/PWA'
import { usePWA, useOfflineStorage } from './hooks/usePWA'
import './App.css'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      cacheTime: 10 * 60 * 1000, // 10 minutes
      retry: 2,
      refetchOnWindowFocus: false,
      // Enable offline support
      networkMode: 'offlineFirst',
    },
    mutations: {
      // Retry mutations when back online
      networkMode: 'offlineFirst',
    },
  },
})

function App() {
  const { isInstallable, isUpdateAvailable, isOnline } = usePWA()
  const { saveOfflineData, getOfflineData } = useOfflineStorage()
  const [showInstallPrompt, setShowInstallPrompt] = useState(false)
  const [showUpdatePrompt, setShowUpdatePrompt] = useState(false)

  // Show install prompt after 30 seconds if installable
  useEffect(() => {
    if (isInstallable) {
      const timer = setTimeout(() => {
        const hasShownBefore = localStorage.getItem('pwa-install-prompt-shown')
        if (!hasShownBefore) {
          setShowInstallPrompt(true)
          localStorage.setItem('pwa-install-prompt-shown', 'true')
        }
      }, 30000) // 30 seconds

      return () => clearTimeout(timer)
    }
  }, [isInstallable])

  // Show update prompt when available
  useEffect(() => {
    if (isUpdateAvailable) {
      setShowUpdatePrompt(true)
    }
  }, [isUpdateAvailable])

  // Setup offline data persistence for React Query
  useEffect(() => {
    const persistor = {
      persistClient: async (client: any) => {
        if (!isOnline) {
          await saveOfflineData('react-query-cache', client.getQueryCache().getAll())
        }
      },
      restoreClient: async () => {
        const cachedData = getOfflineData('react-query-cache')
        if (cachedData) {
          return cachedData
        }
        return undefined
      },
      removeClient: async () => {
        localStorage.removeItem('offline_react-query-cache')
      },
    }

    // Restore cache on app start
    persistor.restoreClient().then((cachedData) => {
      if (cachedData) {
        console.log('Restored offline cache')
      }
    })

    // Persist cache when going offline
    const handleOffline = () => {
      persistor.persistClient(queryClient)
    }

    window.addEventListener('offline', handleOffline)
    return () => window.removeEventListener('offline', handleOffline)
  }, [isOnline, saveOfflineData, getOfflineData])

  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <AuthProvider>
          <Router>
            <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
              {/* Offline Indicator */}
              <OfflineIndicator />
              
              <Routes>
                <Route path="/login" element={<Login />} />
                <Route path="/register" element={<Register />} />
                <Route path="/*" element={
                  <ProtectedRoute>
                    <Layout>
                      <Routes>
                        <Route path="/" element={<Dashboard />} />
                        <Route path="/dashboard" element={<Dashboard />} />
                        <Route path="/players" element={<Players />} />
                        <Route path="/teams" element={<Teams />} />
                        <Route path="/predictions" element={<Predictions />} />
                        <Route path="/analytics" element={<Analytics />} />
                      </Routes>
                    </Layout>
                  </ProtectedRoute>
                } />
              </Routes>
            </div>
          </Router>
          
          {/* PWA Prompts */}
          <PWAInstallPrompt 
            isVisible={showInstallPrompt}
            onClose={() => setShowInstallPrompt(false)}
          />
          
          <PWAUpdatePrompt 
            isVisible={showUpdatePrompt}
            onClose={() => setShowUpdatePrompt(false)}
          />
          
          <Toaster
            position="top-right"
            toastOptions={{
              duration: 4000,
              style: {
                background: 'var(--toast-bg)',
                color: 'var(--toast-color)',
                border: '1px solid var(--toast-border)',
              },
              success: {
                iconTheme: {
                  primary: '#10b981',
                  secondary: '#ffffff',
                },
              },
              error: {
                iconTheme: {
                  primary: '#ef4444',
                  secondary: '#ffffff',
                },
              },
            }}
          />
        </AuthProvider>
      </ThemeProvider>
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  )
}

export default App