#!/usr/bin/env python

# Copyright (c) 2024 German Aerospace Center (DLR), Institute for Software Technology, Germany
# SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import re
import ast

def write(file, *args):
    file.write('    '.join((str(arg) for arg in args)) + '\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=argparse.FileType('r'))
    args = parser.parse_args()

    firstSweepPattern = re.compile('^Initial residual norm: (.*) \(abs\), (.*) \(rel\), ranks: (.*)$')
    sweepPattern = re.compile('^Sweep (.*) residual norm: (.*) \(abs\), (.*) \(rel\), ranks: (.*)$')
    firstTTGmresPattern = re.compile('^ \(M\)ALS local problem: Initial residual norm: (.*) \(abs\), (.*) \(rel\), rhs norm: (.*), ranks: (.*)$')
    ttGmresPattern = re.compile('^ \(M\)ALS local problem: TT-GMRES iteration (.*) residual norm: (.*) \(abs\), (.*) \(rel\), ranks: (.*)$')
    firstGmresPattern = re.compile('^ \(M\)ALS local problem: Initial residual norm: (.*), rhs norm: (.*)$')
    gmresPattern = re.compile('^ \(M\)ALS local problem: GMRES iteration (.*) residual: (.*)$')

    print('Reading file', args.file.name)
    iterLines = (line for line in args.file if 'residual norm:' in line or 'residual:' in line)

    # parse to gnuplot format
    with open(args.file.name + '.dat', 'w') as datFile, open(args.file.name + '.extracted', 'w') as extractFile:
        print('Writing extracted iteration log to:', extractFile.name)
        print('Parsing iteration log to:', datFile.name)

        # first pass: get inner iterations
        write(datFile, '# Inner iterations')
        write(datFile, '# outerIter  totalInnerIter  innerIter   innerResNorm    subspaceMaxRank')

        innerIter = 0
        totalInnerIter = 0
        outerIter = 0

        sweepData = []
        for line in iterLines:
            extractFile.write(line)

            if matchTTGmres := ttGmresPattern.match(line):
                assert(outerIter > 0)
                innerIter += 1
                assert(innerIter == int(matchTTGmres.group(1)))
                totalInnerIter += 1
                innerResNorm = float(matchTTGmres.group(2))
                innerRanks = ast.literal_eval(matchTTGmres.group(4))
                write(datFile, outerIter, totalInnerIter, innerIter, innerResNorm, max(innerRanks))

            elif matchGmres := gmresPattern.match(line):
                assert(outerIter > 0)
                innerIter += 1
                assert(innerIter == int(matchGmres.group(1)))
                totalInnerIter += 1
                innerResNorm = float(matchGmres.group(2))
                write(datFile, outerIter, totalInnerIter, innerIter, innerResNorm)

            elif matchFirstTTGmres := firstTTGmresPattern.match(line):
                assert(outerIter > 0)
                innerIter = 0
                innerResNorm = float(matchFirstTTGmres.group(1))
                innerRanks = ast.literal_eval(matchFirstTTGmres.group(4))
                if totalInnerIter > 0:
                    write(datFile, '')
                write(datFile, outerIter, totalInnerIter, innerIter, innerResNorm, max(innerRanks))

            elif matchFirstGmres := firstGmresPattern.match(line):
                assert(outerIter > 0)
                innerIter = 0
                innerResNorm = float(matchFirstGmres.group(1))
                if totalInnerIter > 0:
                    write(datFile, '')
                write(datFile, outerIter, totalInnerIter, innerIter, innerResNorm)

            elif matchSweep := sweepPattern.match(line):
                #assert(outerIter == float(matchSweep.group(1)))
                resNorm = float(matchSweep.group(2))
                ranks = ast.literal_eval(matchSweep.group(4))
                sweepData += [(outerIter, totalInnerIter, resNorm, max(ranks))]
                outerIter += 0.5

            elif matchFirstSweep := firstSweepPattern.match(line):
                #assert(totalInnerIter == 0)
                #assert(outerIter == 0)
                resNorm = min(1., float(matchFirstSweep.group(1)))
                ranks = ast.literal_eval(matchFirstSweep.group(3))
                sweepData += [(outerIter, totalInnerIter, resNorm, max(ranks))]
                outerIter += 0.5

            else:
                print('Warning: unmatched line:', line)


        
        # also output sweep data
        write(datFile, '')
        write(datFile, '')
        write(datFile, '# Outer iterations')
        write(datFile, '# outerIter  totalInnerIter  resNorm maxRank')
        for sweep in sweepData:
            write(datFile, *sweep)


if __name__ == '__main__':
    main()
