import { useState, useEffect, useCallback } from 'react'
import { toast } from 'react-hot-toast'

interface PWAState {
  isInstallable: boolean
  isInstalled: boolean
  isOnline: boolean
  isUpdateAvailable: boolean
  isLoading: boolean
}

interface PWAActions {
  installApp: () => Promise<void>
  updateApp: () => Promise<void>
  shareContent: (data: ShareData) => Promise<void>
  requestNotificationPermission: () => Promise<NotificationPermission>
  showNotification: (title: string, options?: NotificationOptions) => Promise<void>
}

interface UsePWAReturn extends PWAState, PWAActions {}

let deferredPrompt: any = null
let swRegistration: ServiceWorkerRegistration | null = null

export const usePWA = (): UsePWAReturn => {
  const [state, setState] = useState<PWAState>({
    isInstallable: false,
    isInstalled: false,
    isOnline: navigator.onLine,
    isUpdateAvailable: false,
    isLoading: true
  })

  // Check if app is installed (PWA)
  const checkInstallStatus = useCallback(() => {
    const isInstalled = 
      window.matchMedia('(display-mode: standalone)').matches ||
      (window.navigator as any).standalone ||
      document.referrer.includes('android-app://')
    
    setState(prev => ({ ...prev, isInstalled }))
  }, [])

  // Register service worker
  const registerServiceWorker = useCallback(async () => {
    if ('serviceWorker' in navigator) {
      try {
        const registration = await navigator.serviceWorker.register('/sw.js')
        swRegistration = registration
        
        console.log('Service Worker registered:', registration)
        
        // Check for updates
        registration.addEventListener('updatefound', () => {
          const newWorker = registration.installing
          if (newWorker) {
            newWorker.addEventListener('statechange', () => {
              if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
                setState(prev => ({ ...prev, isUpdateAvailable: true }))
                toast('Nova versão disponível!', {
                  icon: '🔄',
                  duration: 5000
                })
              }
            })
          }
        })
        
        // Listen for messages from service worker
        navigator.serviceWorker.addEventListener('message', (event) => {
          if (event.data?.type === 'SYNC_SUCCESS') {
            toast.success(`Dados sincronizados: ${event.data.data}`)
          }
        })
        
        setState(prev => ({ ...prev, isLoading: false }))
      } catch (error) {
        console.error('Service Worker registration failed:', error)
        setState(prev => ({ ...prev, isLoading: false }))
      }
    } else {
      setState(prev => ({ ...prev, isLoading: false }))
    }
  }, [])

  // Handle install prompt
  const handleInstallPrompt = useCallback((e: Event) => {
    e.preventDefault()
    deferredPrompt = e
    setState(prev => ({ ...prev, isInstallable: true }))
  }, [])

  // Handle online/offline status
  const handleOnlineStatus = useCallback(() => {
    const isOnline = navigator.onLine
    setState(prev => ({ ...prev, isOnline }))
    
    if (isOnline) {
      toast.success('Conexão restaurada!')
      // Trigger background sync if available
      if (swRegistration?.sync) {
        swRegistration.sync.register('sync-predictions')
        swRegistration.sync.register('sync-analytics')
      }
    } else {
      toast.error('Você está offline')
    }
  }, [])

  // Install app
  const installApp = useCallback(async () => {
    if (!deferredPrompt) {
      toast.error('Instalação não disponível')
      return
    }

    try {
      deferredPrompt.prompt()
      const { outcome } = await deferredPrompt.userChoice
      
      if (outcome === 'accepted') {
        toast.success('App instalado com sucesso!')
        setState(prev => ({ ...prev, isInstallable: false, isInstalled: true }))
      } else {
        toast('Instalação cancelada')
      }
      
      deferredPrompt = null
    } catch (error) {
      console.error('Install failed:', error)
      toast.error('Falha na instalação')
    }
  }, [])

  // Update app
  const updateApp = useCallback(async () => {
    if (!swRegistration) {
      toast.error('Service Worker não disponível')
      return
    }

    try {
      const newWorker = swRegistration.waiting
      if (newWorker) {
        newWorker.postMessage({ type: 'SKIP_WAITING' })
        
        // Wait for the new service worker to take control
        navigator.serviceWorker.addEventListener('controllerchange', () => {
          window.location.reload()
        })
        
        toast.success('Atualizando aplicação...')
      }
    } catch (error) {
      console.error('Update failed:', error)
      toast.error('Falha na atualização')
    }
  }, [])

  // Share content
  const shareContent = useCallback(async (data: ShareData) => {
    if (navigator.share) {
      try {
        await navigator.share(data)
        toast.success('Conteúdo compartilhado!')
      } catch (error) {
        if ((error as Error).name !== 'AbortError') {
          console.error('Share failed:', error)
          toast.error('Falha ao compartilhar')
        }
      }
    } else {
      // Fallback to clipboard
      try {
        await navigator.clipboard.writeText(data.url || data.text || '')
        toast.success('Link copiado para a área de transferência!')
      } catch (error) {
        console.error('Clipboard failed:', error)
        toast.error('Falha ao copiar link')
      }
    }
  }, [])

  // Request notification permission
  const requestNotificationPermission = useCallback(async (): Promise<NotificationPermission> => {
    if (!('Notification' in window)) {
      toast.error('Notificações não suportadas')
      return 'denied'
    }

    if (Notification.permission === 'granted') {
      return 'granted'
    }

    if (Notification.permission === 'denied') {
      toast.error('Notificações bloqueadas pelo usuário')
      return 'denied'
    }

    try {
      const permission = await Notification.requestPermission()
      
      if (permission === 'granted') {
        toast.success('Notificações ativadas!')
      } else {
        toast.error('Notificações negadas')
      }
      
      return permission
    } catch (error) {
      console.error('Notification permission failed:', error)
      toast.error('Falha ao solicitar permissão')
      return 'denied'
    }
  }, [])

  // Show notification
  const showNotification = useCallback(async (
    title: string, 
    options?: NotificationOptions
  ) => {
    if (Notification.permission !== 'granted') {
      const permission = await requestNotificationPermission()
      if (permission !== 'granted') {
        return
      }
    }

    try {
      if (swRegistration) {
        // Use service worker for better reliability
        await swRegistration.showNotification(title, {
          icon: '/icons/icon-192x192.png',
          badge: '/icons/icon-72x72.png',
          vibrate: [200, 100, 200],
          ...options
        })
      } else {
        // Fallback to regular notification
        new Notification(title, {
          icon: '/icons/icon-192x192.png',
          ...options
        })
      }
    } catch (error) {
      console.error('Show notification failed:', error)
      toast.error('Falha ao exibir notificação')
    }
  }, [requestNotificationPermission])

  // Setup event listeners
  useEffect(() => {
    registerServiceWorker()
    checkInstallStatus()

    // Install prompt
    window.addEventListener('beforeinstallprompt', handleInstallPrompt)
    
    // Online/offline status
    window.addEventListener('online', handleOnlineStatus)
    window.addEventListener('offline', handleOnlineStatus)
    
    // App installed
    window.addEventListener('appinstalled', () => {
      setState(prev => ({ ...prev, isInstalled: true, isInstallable: false }))
      toast.success('App instalado com sucesso!')
    })

    return () => {
      window.removeEventListener('beforeinstallprompt', handleInstallPrompt)
      window.removeEventListener('online', handleOnlineStatus)
      window.removeEventListener('offline', handleOnlineStatus)
    }
  }, [registerServiceWorker, checkInstallStatus, handleInstallPrompt, handleOnlineStatus])

  return {
    ...state,
    installApp,
    updateApp,
    shareContent,
    requestNotificationPermission,
    showNotification
  }
}

// Hook for background sync
export const useBackgroundSync = () => {
  const registerSync = useCallback(async (tag: string) => {
    if ('serviceWorker' in navigator && 'sync' in window.ServiceWorkerRegistration.prototype) {
      try {
        const registration = await navigator.serviceWorker.ready
        await registration.sync.register(tag)
        console.log('Background sync registered:', tag)
      } catch (error) {
        console.error('Background sync registration failed:', error)
      }
    }
  }, [])

  return { registerSync }
}

// Hook for offline storage
export const useOfflineStorage = () => {
  const saveOfflineData = useCallback(async (key: string, data: any) => {
    try {
      const serializedData = JSON.stringify({
        data,
        timestamp: Date.now(),
        version: '1.0.0'
      })
      
      localStorage.setItem(`offline_${key}`, serializedData)
      console.log('Data saved offline:', key)
    } catch (error) {
      console.error('Failed to save offline data:', error)
    }
  }, [])

  const getOfflineData = useCallback((key: string) => {
    try {
      const serializedData = localStorage.getItem(`offline_${key}`)
      if (!serializedData) return null
      
      const { data, timestamp, version } = JSON.parse(serializedData)
      
      // Check if data is not too old (24 hours)
      const isExpired = Date.now() - timestamp > 24 * 60 * 60 * 1000
      if (isExpired) {
        localStorage.removeItem(`offline_${key}`)
        return null
      }
      
      return data
    } catch (error) {
      console.error('Failed to get offline data:', error)
      return null
    }
  }, [])

  const clearOfflineData = useCallback((key?: string) => {
    try {
      if (key) {
        localStorage.removeItem(`offline_${key}`)
      } else {
        // Clear all offline data
        Object.keys(localStorage)
          .filter(k => k.startsWith('offline_'))
          .forEach(k => localStorage.removeItem(k))
      }
    } catch (error) {
      console.error('Failed to clear offline data:', error)
    }
  }, [])

  return {
    saveOfflineData,
    getOfflineData,
    clearOfflineData
  }
}