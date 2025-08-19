import { motion } from 'framer-motion'
import { Calendar, Clock, MapPin } from 'lucide-react'
import { format } from 'date-fns'
import { ptBR } from 'date-fns/locale'

interface Match {
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
}

interface RecentMatchesProps {
  matches: Match[]
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

export default function RecentMatches({ matches }: RecentMatchesProps) {
  const formatMatchDate = (dateString: string) => {
    try {
      const date = new Date(dateString)
      return format(date, "dd/MM 'às' HH:mm", { locale: ptBR })
    } catch {
      return dateString
    }
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            Partidas Recentes
          </h3>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
            Últimas partidas analisadas
          </p>
        </div>
        <div className="p-2 bg-green-50 dark:bg-green-900/20 rounded-lg">
          <Calendar className="h-5 w-5 text-green-600 dark:text-green-400" />
        </div>
      </div>

      {/* Matches List */}
      <div className="space-y-4">
        {matches && matches.length > 0 ? (
          matches.slice(0, 5).map((match, index) => (
            <motion.div
              key={match.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
              className="p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
            >
              {/* Match Header */}
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center space-x-2">
                  <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                    statusColors[match.status]
                  }`}>
                    {statusLabels[match.status]}
                  </span>
                  {match.status === 'live' && (
                    <div className="flex items-center space-x-1 text-red-600 dark:text-red-400">
                      <div className="w-2 h-2 bg-red-600 rounded-full animate-pulse"></div>
                      <Clock className="h-3 w-3" />
                    </div>
                  )}
                </div>
                <div className="flex items-center text-xs text-gray-500 dark:text-gray-400">
                  <Calendar className="h-3 w-3 mr-1" />
                  {formatMatchDate(match.date)}
                </div>
              </div>

              {/* Teams and Score */}
              <div className="flex items-center justify-between mb-3">
                <div className="flex-1">
                  <div className="flex items-center justify-between">
                    <span className="font-medium text-gray-900 dark:text-white">
                      {match.homeTeam}
                    </span>
                    {match.status === 'finished' && match.homeScore !== undefined && (
                      <span className="text-lg font-bold text-gray-900 dark:text-white">
                        {match.homeScore}
                      </span>
                    )}
                  </div>
                  <div className="flex items-center justify-between mt-1">
                    <span className="font-medium text-gray-900 dark:text-white">
                      {match.awayTeam}
                    </span>
                    {match.status === 'finished' && match.awayScore !== undefined && (
                      <span className="text-lg font-bold text-gray-900 dark:text-white">
                        {match.awayScore}
                      </span>
                    )}
                  </div>
                </div>
              </div>

              {/* Stats */}
              {(match.foulsHome !== undefined || match.cardsHome !== undefined) && (
                <div className="grid grid-cols-2 gap-4 text-xs">
                  {match.foulsHome !== undefined && (
                    <div>
                      <span className="text-gray-500 dark:text-gray-400">Faltas:</span>
                      <div className="flex justify-between mt-1">
                        <span className="text-gray-900 dark:text-white">{match.foulsHome}</span>
                        <span className="text-gray-900 dark:text-white">{match.foulsAway || 0}</span>
                      </div>
                    </div>
                  )}
                  {match.cardsHome !== undefined && (
                    <div>
                      <span className="text-gray-500 dark:text-gray-400">Cartões:</span>
                      <div className="flex justify-between mt-1">
                        <span className="text-gray-900 dark:text-white">{match.cardsHome}</span>
                        <span className="text-gray-900 dark:text-white">{match.cardsAway || 0}</span>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Venue */}
              {match.venue && (
                <div className="flex items-center mt-2 text-xs text-gray-500 dark:text-gray-400">
                  <MapPin className="h-3 w-3 mr-1" />
                  {match.venue}
                </div>
              )}
            </motion.div>
          ))
        ) : (
          <div className="text-center py-8">
            <Calendar className="h-12 w-12 mx-auto text-gray-400 dark:text-gray-600 mb-3" />
            <p className="text-gray-500 dark:text-gray-400">
              Nenhuma partida encontrada
            </p>
          </div>
        )}
      </div>

      {matches && matches.length > 5 && (
        <div className="mt-4 text-center">
          <button className="text-sm text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 font-medium">
            Ver todas as partidas
          </button>
        </div>
      )}
    </div>
  )
}