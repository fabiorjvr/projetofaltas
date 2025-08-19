import { motion } from 'framer-motion'
import { User, Trophy, AlertTriangle, Shield } from 'lucide-react'
import { clsx } from 'clsx'

interface Player {
  id: string
  name: string
  team: string
  position: string
  fouls: number
  cards: number
  matches: number
  avatar?: string
  riskLevel: 'low' | 'medium' | 'high'
}

interface TopPlayersProps {
  players: Player[]
}

const riskColors = {
  low: 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400',
  medium: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400',
  high: 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400'
}

const riskLabels = {
  low: 'Baixo',
  medium: 'Médio',
  high: 'Alto'
}

export default function TopPlayers({ players }: TopPlayersProps) {
  const getPlayerAvatar = (player: Player) => {
    if (player.avatar) {
      return (
        <img
          src={player.avatar}
          alt={player.name}
          className="w-10 h-10 rounded-full object-cover"
        />
      )
    }
    
    return (
      <div className="w-10 h-10 bg-gray-200 dark:bg-gray-600 rounded-full flex items-center justify-center">
        <User className="h-5 w-5 text-gray-500 dark:text-gray-400" />
      </div>
    )
  }

  const calculateFoulsPerMatch = (fouls: number, matches: number) => {
    if (matches === 0) return 0
    return (fouls / matches).toFixed(1)
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            Top Jogadores
          </h3>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
            Jogadores com mais faltas
          </p>
        </div>
        <div className="p-2 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
          <Trophy className="h-5 w-5 text-purple-600 dark:text-purple-400" />
        </div>
      </div>

      {/* Players List */}
      <div className="space-y-4">
        {players && players.length > 0 ? (
          players.slice(0, 5).map((player, index) => (
            <motion.div
              key={player.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
              className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
            >
              {/* Player Info */}
              <div className="flex items-center space-x-3 flex-1">
                {/* Ranking */}
                <div className={clsx(
                  'w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold',
                  index === 0 ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400' :
                  index === 1 ? 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300' :
                  index === 2 ? 'bg-orange-100 text-orange-800 dark:bg-orange-900/20 dark:text-orange-400' :
                  'bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400'
                )}>
                  {index + 1}
                </div>

                {/* Avatar */}
                {getPlayerAvatar(player)}

                {/* Details */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center space-x-2">
                    <h4 className="font-medium text-gray-900 dark:text-white truncate">
                      {player.name}
                    </h4>
                    <span className={clsx(
                      'px-2 py-1 text-xs font-medium rounded-full',
                      riskColors[player.riskLevel]
                    )}>
                      {riskLabels[player.riskLevel]}
                    </span>
                  </div>
                  <div className="flex items-center space-x-4 mt-1 text-xs text-gray-500 dark:text-gray-400">
                    <span>{player.team}</span>
                    <span>•</span>
                    <span>{player.position}</span>
                    <span>•</span>
                    <span>{player.matches} jogos</span>
                  </div>
                </div>
              </div>

              {/* Stats */}
              <div className="flex items-center space-x-4 text-sm">
                <div className="text-center">
                  <div className="flex items-center space-x-1 text-yellow-600 dark:text-yellow-400">
                    <AlertTriangle className="h-4 w-4" />
                    <span className="font-medium">{player.fouls}</span>
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    {calculateFoulsPerMatch(player.fouls, player.matches)}/jogo
                  </div>
                </div>
                
                <div className="text-center">
                  <div className="flex items-center space-x-1 text-red-600 dark:text-red-400">
                    <Shield className="h-4 w-4" />
                    <span className="font-medium">{player.cards}</span>
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    cartões
                  </div>
                </div>
              </div>
            </motion.div>
          ))
        ) : (
          <div className="text-center py-8">
            <User className="h-12 w-12 mx-auto text-gray-400 dark:text-gray-600 mb-3" />
            <p className="text-gray-500 dark:text-gray-400">
              Nenhum jogador encontrado
            </p>
          </div>
        )}
      </div>

      {players && players.length > 5 && (
        <div className="mt-4 text-center">
          <button className="text-sm text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 font-medium">
            Ver todos os jogadores
          </button>
        </div>
      )}
    </div>
  )
}