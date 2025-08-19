const CACHE_NAME = 'football-fouls-analytics-v1.0.0'
const STATIC_CACHE_NAME = 'static-v1.0.0'
const DYNAMIC_CACHE_NAME = 'dynamic-v1.0.0'
const API_CACHE_NAME = 'api-v1.0.0'

// Assets to cache immediately
const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/manifest.json',
  '/static/js/bundle.js',
  '/static/css/main.css',
  '/icons/icon-192x192.png',
  '/icons/icon-512x512.png'
]

// API endpoints to cache
const API_ENDPOINTS = [
  '/api/dashboard/stats',
  '/api/players',
  '/api/teams',
  '/api/predictions'
]

// Install event - cache static assets
self.addEventListener('install', event => {
  console.log('[SW] Installing service worker...')
  
  event.waitUntil(
    Promise.all([
      caches.open(STATIC_CACHE_NAME).then(cache => {
        console.log('[SW] Caching static assets')
        return cache.addAll(STATIC_ASSETS)
      }),
      self.skipWaiting()
    ])
  )
})

// Activate event - clean up old caches
self.addEventListener('activate', event => {
  console.log('[SW] Activating service worker...')
  
  event.waitUntil(
    Promise.all([
      caches.keys().then(cacheNames => {
        return Promise.all(
          cacheNames.map(cacheName => {
            if (cacheName !== STATIC_CACHE_NAME && 
                cacheName !== DYNAMIC_CACHE_NAME && 
                cacheName !== API_CACHE_NAME) {
              console.log('[SW] Deleting old cache:', cacheName)
              return caches.delete(cacheName)
            }
          })
        )
      }),
      self.clients.claim()
    ])
  )
})

// Fetch event - handle requests with different strategies
self.addEventListener('fetch', event => {
  const { request } = event
  const url = new URL(request.url)

  // Skip non-GET requests
  if (request.method !== 'GET') {
    return
  }

  // Handle different types of requests
  if (url.pathname.startsWith('/api/')) {
    // API requests - Network First with fallback to cache
    event.respondWith(handleApiRequest(request))
  } else if (url.pathname.match(/\.(js|css|png|jpg|jpeg|svg|ico|woff|woff2)$/)) {
    // Static assets - Cache First
    event.respondWith(handleStaticAssets(request))
  } else {
    // HTML pages - Network First with fallback
    event.respondWith(handlePageRequest(request))
  }
})

// Network First strategy for API requests
async function handleApiRequest(request) {
  const cache = await caches.open(API_CACHE_NAME)
  
  try {
    // Try network first
    const networkResponse = await fetch(request)
    
    if (networkResponse.ok) {
      // Cache successful responses
      cache.put(request, networkResponse.clone())
      return networkResponse
    }
    
    // If network fails, try cache
    const cachedResponse = await cache.match(request)
    if (cachedResponse) {
      console.log('[SW] Serving API from cache:', request.url)
      return cachedResponse
    }
    
    // Return network response even if not ok
    return networkResponse
  } catch (error) {
    console.log('[SW] Network failed, trying cache for:', request.url)
    
    // Network failed, try cache
    const cachedResponse = await cache.match(request)
    if (cachedResponse) {
      return cachedResponse
    }
    
    // Return offline fallback for API
    return new Response(
      JSON.stringify({ 
        error: 'Offline', 
        message: 'Dados não disponíveis offline' 
      }),
      {
        status: 503,
        statusText: 'Service Unavailable',
        headers: { 'Content-Type': 'application/json' }
      }
    )
  }
}

// Cache First strategy for static assets
async function handleStaticAssets(request) {
  const cache = await caches.open(STATIC_CACHE_NAME)
  
  // Try cache first
  const cachedResponse = await cache.match(request)
  if (cachedResponse) {
    return cachedResponse
  }
  
  // If not in cache, fetch from network and cache
  try {
    const networkResponse = await fetch(request)
    if (networkResponse.ok) {
      cache.put(request, networkResponse.clone())
    }
    return networkResponse
  } catch (error) {
    console.log('[SW] Failed to fetch static asset:', request.url)
    throw error
  }
}

// Network First strategy for HTML pages
async function handlePageRequest(request) {
  const cache = await caches.open(DYNAMIC_CACHE_NAME)
  
  try {
    // Try network first
    const networkResponse = await fetch(request)
    
    if (networkResponse.ok) {
      // Cache successful responses
      cache.put(request, networkResponse.clone())
      return networkResponse
    }
    
    // If network response is not ok, try cache
    const cachedResponse = await cache.match(request)
    if (cachedResponse) {
      return cachedResponse
    }
    
    return networkResponse
  } catch (error) {
    console.log('[SW] Network failed for page:', request.url)
    
    // Network failed, try cache
    const cachedResponse = await cache.match(request)
    if (cachedResponse) {
      return cachedResponse
    }
    
    // Return offline fallback page
    const offlinePage = await cache.match('/')
    if (offlinePage) {
      return offlinePage
    }
    
    // Last resort - basic offline response
    return new Response(
      `<!DOCTYPE html>
      <html>
        <head>
          <title>Offline - Football Fouls Analytics</title>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <style>
            body { 
              font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
              display: flex; 
              align-items: center; 
              justify-content: center; 
              min-height: 100vh; 
              margin: 0;
              background: #f3f4f6;
              color: #374151;
            }
            .container {
              text-align: center;
              padding: 2rem;
              background: white;
              border-radius: 8px;
              box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            }
            .icon { font-size: 4rem; margin-bottom: 1rem; }
            h1 { margin: 0 0 1rem 0; color: #1f2937; }
            p { margin: 0; color: #6b7280; }
          </style>
        </head>
        <body>
          <div class="container">
            <div class="icon">⚽</div>
            <h1>Você está offline</h1>
            <p>Verifique sua conexão com a internet e tente novamente.</p>
          </div>
        </body>
      </html>`,
      {
        status: 200,
        statusText: 'OK',
        headers: { 'Content-Type': 'text/html' }
      }
    )
  }
}

// Background sync for offline actions
self.addEventListener('sync', event => {
  console.log('[SW] Background sync triggered:', event.tag)
  
  if (event.tag === 'sync-predictions') {
    event.waitUntil(syncPredictions())
  }
  
  if (event.tag === 'sync-analytics') {
    event.waitUntil(syncAnalytics())
  }
})

// Sync predictions when back online
async function syncPredictions() {
  try {
    console.log('[SW] Syncing predictions...')
    const response = await fetch('/api/predictions/sync', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    })
    
    if (response.ok) {
      console.log('[SW] Predictions synced successfully')
      // Notify clients about successful sync
      self.clients.matchAll().then(clients => {
        clients.forEach(client => {
          client.postMessage({
            type: 'SYNC_SUCCESS',
            data: 'predictions'
          })
        })
      })
    }
  } catch (error) {
    console.error('[SW] Failed to sync predictions:', error)
  }
}

// Sync analytics when back online
async function syncAnalytics() {
  try {
    console.log('[SW] Syncing analytics...')
    const response = await fetch('/api/analytics/sync', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    })
    
    if (response.ok) {
      console.log('[SW] Analytics synced successfully')
      // Notify clients about successful sync
      self.clients.matchAll().then(clients => {
        clients.forEach(client => {
          client.postMessage({
            type: 'SYNC_SUCCESS',
            data: 'analytics'
          })
        })
      })
    }
  } catch (error) {
    console.error('[SW] Failed to sync analytics:', error)
  }
}

// Push notifications
self.addEventListener('push', event => {
  console.log('[SW] Push notification received')
  
  const options = {
    body: 'Nova predição de alto risco disponível!',
    icon: '/icons/icon-192x192.png',
    badge: '/icons/icon-72x72.png',
    vibrate: [200, 100, 200],
    data: {
      url: '/predictions'
    },
    actions: [
      {
        action: 'view',
        title: 'Ver Predições',
        icon: '/icons/icon-96x96.png'
      },
      {
        action: 'dismiss',
        title: 'Dispensar'
      }
    ]
  }
  
  if (event.data) {
    const data = event.data.json()
    options.body = data.message || options.body
    options.data = { ...options.data, ...data }
  }
  
  event.waitUntil(
    self.registration.showNotification('Football Fouls Analytics', options)
  )
})

// Handle notification clicks
self.addEventListener('notificationclick', event => {
  console.log('[SW] Notification clicked:', event.action)
  
  event.notification.close()
  
  if (event.action === 'view') {
    const url = event.notification.data?.url || '/'
    event.waitUntil(
      self.clients.matchAll({ type: 'window' }).then(clients => {
        // Check if there's already a window open
        for (const client of clients) {
          if (client.url.includes(url) && 'focus' in client) {
            return client.focus()
          }
        }
        
        // Open new window
        if (self.clients.openWindow) {
          return self.clients.openWindow(url)
        }
      })
    )
  }
})

// Message handling from main thread
self.addEventListener('message', event => {
  console.log('[SW] Message received:', event.data)
  
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting()
  }
  
  if (event.data && event.data.type === 'CACHE_URLS') {
    event.waitUntil(
      caches.open(DYNAMIC_CACHE_NAME).then(cache => {
        return cache.addAll(event.data.urls)
      })
    )
  }
})