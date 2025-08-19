import { motion } from 'framer-motion'
import { LucideIcon, TrendingUp, TrendingDown } from 'lucide-react'
import { clsx } from 'clsx'

interface StatsCardProps {
  title: string
  value: number | string
  icon: LucideIcon
  trend?: number
  color: 'blue' | 'green' | 'yellow' | 'purple' | 'red'
  subtitle?: string
}

const colorClasses = {
  blue: {
    bg: 'bg-blue-50 dark:bg-blue-900/20',
    icon: 'text-blue-600 dark:text-blue-400',
    trend: 'text-blue-600 dark:text-blue-400'
  },
  green: {
    bg: 'bg-green-50 dark:bg-green-900/20',
    icon: 'text-green-600 dark:text-green-400',
    trend: 'text-green-600 dark:text-green-400'
  },
  yellow: {
    bg: 'bg-yellow-50 dark:bg-yellow-900/20',
    icon: 'text-yellow-600 dark:text-yellow-400',
    trend: 'text-yellow-600 dark:text-yellow-400'
  },
  purple: {
    bg: 'bg-purple-50 dark:bg-purple-900/20',
    icon: 'text-purple-600 dark:text-purple-400',
    trend: 'text-purple-600 dark:text-purple-400'
  },
  red: {
    bg: 'bg-red-50 dark:bg-red-900/20',
    icon: 'text-red-600 dark:text-red-400',
    trend: 'text-red-600 dark:text-red-400'
  }
}

export default function StatsCard({ 
  title, 
  value, 
  icon: Icon, 
  trend, 
  color, 
  subtitle 
}: StatsCardProps) {
  const colors = colorClasses[color]
  const isPositiveTrend = trend && trend > 0
  const isNegativeTrend = trend && trend < 0

  const formatValue = (val: number | string) => {
    if (typeof val === 'number') {
      return val.toLocaleString('pt-BR')
    }
    return val
  }

  return (
    <motion.div
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6 hover:shadow-md transition-shadow"
    >
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
            {title}
          </p>
          <p className="text-2xl font-bold text-gray-900 dark:text-white mt-1">
            {formatValue(value)}
          </p>
          {subtitle && (
            <p className="text-xs text-gray-500 dark:text-gray-500 mt-1">
              {subtitle}
            </p>
          )}
        </div>
        
        <div className={clsx(
          'p-3 rounded-lg',
          colors.bg
        )}>
          <Icon className={clsx('h-6 w-6', colors.icon)} />
        </div>
      </div>
      
      {trend !== undefined && (
        <div className="mt-4 flex items-center">
          <div className={clsx(
            'flex items-center space-x-1 text-sm font-medium',
            isPositiveTrend ? 'text-green-600 dark:text-green-400' : 
            isNegativeTrend ? 'text-red-600 dark:text-red-400' : 
            'text-gray-500 dark:text-gray-400'
          )}>
            {isPositiveTrend && <TrendingUp className="h-4 w-4" />}
            {isNegativeTrend && <TrendingDown className="h-4 w-4" />}
            <span>
              {trend > 0 ? '+' : ''}{trend}%
            </span>
          </div>
          <span className="text-sm text-gray-500 dark:text-gray-400 ml-2">
            vs. mês anterior
          </span>
        </div>
      )}
    </motion.div>
  )
}