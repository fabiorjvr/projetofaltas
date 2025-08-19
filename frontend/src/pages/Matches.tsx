import { useState } from 'react'
import { motion } from 'framer-motion'
import { useQuery } from '@tanstack/react-query'
import { 
  Calendar, 
  Clock, 
  MapPin, 
  Filter,
  AlertTriangle,
  Shield,
  ChevronLeft,
  ChevronRight,
  Play,
  Pause
} from 'lucide-react'
import { format } from 'date-fns'
import { ptBR } from 'date-fns/locale'
import { fetchMatches } from '../services/api'
import type { Match } from '../services/api'

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.05
    }
  }
}

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0 }
}

const statusColors = {
  scheduled: 'bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400',
  live: 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400',
  finished: 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400'
}

const statusLabels = {
  scheduled: 'Agendado',
  live: 'Ao Vivo',
  finished: 'Finalizado'
}

const statusIcons = {
  scheduled: Clock,
  live: Play,
  finished: Pause
}

export default function Matches() {
  const [page, setPage] = useState(1)
  const [statusFilter, setStatusFilter] = useState('')
  const [teamFilter, setTeamFilter] = useState('')
  const [showFilters, setShowFilters] = useState(false)

  const { data, isLoading, error } = useQuery({
    queryKey: ['matches', page, statusFilter, teamFilter],
    queryFn: () => fetchMatches({
      page,
      limit: 10,
      status: statusFilter || undefined,
      team: teamFilter || undefined
    }),
    keepPreviousData: true
  })

  const formatMatchDate = (dateString: string) => {
    try {
      const date = new Date(dateString)
      return format(date, "dd/MM/yyyy 'às' HH:mm", { locale: ptBR })
    } catch {
      return dateString
    }
  }

  const formatMatchTime = (dateString: string) => {
    try {
      const date = new Date(dateString)
      return format(date, 'HH:mm', { locale: ptBR })
    } catch {
      return '--:--'
    }
  }

  const clearFilters = () => {
    setStatusFilter('')
    setTeamFilter('')
    setPage(1)
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <AlertTriangle className="h-12 w-12 text-red-500 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
            Erro ao carregar partidas
          </h3>
          <p className="text-gray-600 dark:text-gray-400">
            Tente novamente em alguns instantes
          </p>
        </div>
      </div>
    )
  }

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="space-y-6"
    >
      {/* Header */}
      <motion.div variants={itemVariants}>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
          Partidas
        </h1>
        <p className="mt-2 text-gray-600 dark:text-gray-400">
          Acompanhe as partidas e suas estatísticas de faltas e cartões
        </p>
      </motion.div>

      {/* Filters */}
      <motion.div variants={itemVariants} className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            Filtros
          </h3>
          <button
            onClick={() => setShowFilters(!showFilters)}
            className="flex items-center space-x-2 px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
          >
            <Filter className="h-5 w-5" />
            <span>{showFilters ? 'Ocultar' : 'Mostrar'} Filtros</span>
          </button>
        </div>

        {showFilters && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4"
          >
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Status
              </label>
              <select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
              >
                <option value="">Todos os status</option>
                <option value="scheduled">Agendado</option>
                <option value="live">Ao Vivo</option>
                <option value="finished">Finalizado</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Time
              </label>
              <select
                value={teamFilter}
                onChange={(e) => setTeamFilter(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
              >
                <option value="">Todos os times</option>
                <option value="Barcelona">Barcelona</option>
                <option value="Real Madrid">Real Madrid</option>
                <option value="Manchester City">Manchester City</option>
                <option value="Liverpool">Liverpool</option>
              </select>
            </div>
            <div className="flex items-end">
              <button
                onClick={clearFilters}
                className="w-full px-4 py-2 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
              >
                Limpar filtros
              </button>
            </div>
          </motion.div>
        )}
      </motion.div>

      {/* Matches List */}
      {isLoading ? (
        <div className="space-y-4">
          {Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6 animate-pulse">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-4">
                  <div className="w-20 h-6 bg-gray-200 dark:bg-gray-600 rounded"></div>
                  <div className="w-24 h-4 bg-gray-200 dark:bg-gray-600 rounded"></div>
                </div>
                <div className="w-32 h-4 bg-gray-200 dark:bg-gray-600 rounded"></div>
              </div>
              <div className="grid grid-cols-3 gap-4 items-center">
                <div className="text-right">
                  <div className="w-24 h-6 bg-gray-200 dark:bg-gray-600 rounded ml-auto mb-2"></div>
                  <div className="w-16 h-4 bg-gray-200 dark:bg-gray-600 rounded ml-auto"></div>
                </div>
                <div className="text-center">
                  <div className="w-16 h-8 bg-gray-200 dark:bg-gray-600 rounded mx-auto"></div>
                </div>
                <div>
                  <div className="w-24 h-6 bg-gray-200 dark:bg-gray-600 rounded mb-2"></div>
                  <div className="w-16 h-4 bg-gray-200 dark:bg-gray-600 rounded"></div>
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <motion.div variants={containerVariants} className="space-y-4">
          {data?.matches.map((match) => {
            const StatusIcon = statusIcons[match.status]
            
            return (
              <motion.div
                key={match.id}
                variants={itemVariants}
                whileHover={{ scale: 1.01 }}
                className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6 hover:shadow-md transition-shadow"
              >
                {/* Match Header */}
                <div className="flex items-center justify-between mb-6">
                  <div className="flex items-center space-x-4">
                    <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
                      statusColors[match.status]
                    }`}>
                      <StatusIcon className="h-4 w-4 mr-1" />
                      {statusLabels[match.status]}
                    </span>
                    {match.status === 'live' && (
                      <div className="flex items-center space-x-1 text-red-600 dark:text-red-400">
                        <div className="w-2 h-2 bg-red-600 rounded-full animate-pulse"></div>
                        <span className="text-sm font-medium">AO VIVO</span>
                      </div>
                    )}
                  </div>
                  <div className="flex items-center text-sm text-gray-500 dark:text-gray-400">
                    <Calendar className="h-4 w-4 mr-1" />
                    {formatMatchDate(match.date)}
                  </div>
                </div>

                {/* Teams and Score */}
                <div className="grid grid-cols-3 gap-4 items-center mb-6">
                  {/* Home Team */}
                  <div className="text-right">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                      {match.homeTeam}
                    </h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      Casa
                    </p>
                  </div>

                  {/* Score/Time */}
                  <div className="text-center">
                    {match.status === 'finished' && match.homeScore !== undefined && match.awayScore !== undefined ? (
                      <div className="text-2xl font-bold text-gray-900 dark:text-white">
                        {match.homeScore} - {match.awayScore}
                      </div>
                    ) : match.status === 'live' ? (
                      <div className="text-lg font-semibold text-red-600 dark:text-red-400">
                        {formatMatchTime(match.date)}
                      </div>
                    ) : (
                      <div className="text-lg font-semibold text-gray-600 dark:text-gray-400">
                        {formatMatchTime(match.date)}
                      </div>
                    )}
                  </div>

                  {/* Away Team */}
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                      {match.awayTeam}
                    </h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      Visitante
                    </p>
                  </div>
                </div>

                {/* Match Info */}
                <div className="flex items-center justify-between text-sm text-gray-600 dark:text-gray-400 mb-4">
                  <div className="flex items-center space-x-4">
                    <div className="flex items-center">
                      <MapPin className="h-4 w-4 mr-1" />
                      <span>{match.venue || 'Local não informado'}</span>
                    </div>
                    <div>
                      <span className="font-medium">{match.league}</span>
                    </div>
                  </div>
                </div>

                {/* Stats */}
                {match.stats && (
                  <div className="grid grid-cols-2 gap-6 pt-4 border-t border-gray-200 dark:border-gray-700">
                    <div>
                      <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Faltas
                      </h4>
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-1 text-yellow-600 dark:text-yellow-400">
                          <AlertTriangle className="h-4 w-4" />
                          <span className="font-semibold">{match.stats.foulsHome}</span>
                        </div>
                        <span className="text-xs text-gray-500 dark:text-gray-400">vs</span>
                        <div className="flex items-center space-x-1 text-yellow-600 dark:text-yellow-400">
                          <span className="font-semibold">{match.stats.foulsAway}</span>
                          <AlertTriangle className="h-4 w-4" />
                        </div>
                      </div>
                    </div>
                    
                    <div>
                      <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Cartões
                      </h4>
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-1 text-red-600 dark:text-red-400">
                          <Shield className="h-4 w-4" />
                          <span className="font-semibold">{match.stats.cardsHome}</span>
                        </div>
                        <span className="text-xs text-gray-500 dark:text-gray-400">vs</span>
                        <div className="flex items-center space-x-1 text-red-600 dark:text-red-400">
                          <span className="font-semibold">{match.stats.cardsAway}</span>
                          <Shield className="h-4 w-4" />
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </motion.div>
            )
          })}
        </motion.div>
      )}

      {/* Pagination */}
      {data && data.totalPages > 1 && (
        <motion.div variants={itemVariants} className="flex items-center justify-between">
          <div className="text-sm text-gray-700 dark:text-gray-300">
            Mostrando {((page - 1) * 10) + 1} a {Math.min(page * 10, data.total)} de {data.total} partidas
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setPage(page - 1)}
              disabled={page === 1}
              className="p-2 rounded-lg border border-gray-300 dark:border-gray-600 disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
            >
              <ChevronLeft className="h-5 w-5" />
            </button>
            <span className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300">
              {page} de {data.totalPages}
            </span>
            <button
              onClick={() => setPage(page + 1)}
              disabled={page === data.totalPages}
              className="p-2 rounded-lg border border-gray-300 dark:border-gray-600 disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
            >
              <ChevronRight className="h-5 w-5" />
            </button>
          </div>
        </motion.div>
      )}

      {/* Empty State */}
      {!isLoading && (!data?.matches || data.matches.length === 0) && (
        <motion.div variants={itemVariants} className="text-center py-12">
          <Calendar className="h-16 w-16 mx-auto text-gray-400 dark:text-gray-600 mb-4" />
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
            Nenhuma partida encontrada
          </h3>
          <p className="text-gray-600 dark:text-gray-400">
            Tente ajustar os filtros ou aguarde novas partidas
          </p>
        </motion.div>
      )}
    </motion.div>
  )
}