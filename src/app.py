import streamlit as st
from frontend.pages.chat import ChatPage

def main():
    chat_page = ChatPage()
    chat_page.run()

if __name__ == "__main__":
    main()