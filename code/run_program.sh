#!/bin/bash

# Time to wait before sending the signal (in seconds)
WAIT_TIME=5

# Parse arguments to remove -time_limit and its value
PROGRAM_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -time)
            WAIT_TIME="$2" # Set the wait time from -time_limit
            PROGRAM_ARGS+=("$1" "$2") # Keep -time for the program
            shift 2 # Skip -time_limit and its value
            ;;
        *)
            PROGRAM_ARGS+=("$1") # Add other arguments to program arguments
            shift # Move to the next argument
            ;;
    esac
done

# Run the C++ program in the background with the remaining arguments
PROGRAM_NAME="./tsp_solver"
"$PROGRAM_NAME" "${PROGRAM_ARGS[@]}" &
PROGRAM_PID=$!

if [ -z "$PROGRAM_PID" ]; then
    echo "Failed to start $PROGRAM_NAME. Exiting."
    exit 1
fi

echo "Program started with PID: $PROGRAM_PID. Waiting $WAIT_TIME seconds to terminate..."
sleep "$WAIT_TIME"

if ps -p "$PROGRAM_PID" > /dev/null; then
    # Send SIGUSR1 signal to the program
    kill -SIGUSR1 "$PROGRAM_PID"
    echo "Signal sent to PID $PROGRAM_PID."
fi
