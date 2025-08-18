#!/usr/bin/env python3
"""
CLI para o projeto Football Fouls Analytics
"""

import click


@click.group()
def cli():
    """Football Fouls Analytics CLI"""
    pass


@cli.command()
def hello():
    """Comando de teste"""
    click.echo("Hello from Football Fouls Analytics!")


if __name__ == "__main__":
    cli()