from trl.environment import TextHistory
from trl.environment.base_environment import Text, is_rich_available, warnings
from .decoder import incremental_decode

if is_rich_available():
    from rich import print
    from rich.text import Text
    from rich.console import Console
from IPython.display import display, HTML
import hashlib


class CustomTextHistory(TextHistory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_color_map = {}

    def get_color_for_token_id(self, token_id):
        """
        Generate a consistent color for each unique token ID.
        """
        # Use a hashing function to ensure a consistent color for the same token ID
        hash_object = hashlib.md5(str(token_id).encode())
        # Take the first 6 characters from the hash to form a color code
        color_code = "#" + hash_object.hexdigest()[:6]
        return color_code

    def show_text(self, show_legend=False):
        """
        Print the text history.
        """
        if not is_rich_available():
            warnings.warn("install rich to display text")
            return

        text = Text(self.text)

        try:
            text.stylize(self.prompt_color, self.text_spans[0][0], self.text_spans[1][0])
        except IndexError:
            text.stylize(self.prompt_color, self.text_spans[0][0])

        for i, (start, end) in enumerate(self.text_spans[1:]):
            if self.system_spans[i + 1]:
                text.stylize(self.system_color, start, end)
            else:
                text.stylize(self.model_color, start, end)
        print(text)
        if show_legend:
            self.show_colour_legend()

    def show_tokens(self, tokenizer, show_legend=False):
        """
        Print the history tokens.
        """
        if not is_rich_available():
            warnings.warn("install rich to display tokens")
            return

        text = Text()
        prompt_end = self.token_spans[0][1]
        for i, (token, mask) in enumerate(zip(self.tokens, self.token_masks)):
            if i < prompt_end:
                text.append(
                    tokenizer.convert_ids_to_tokens(token.item()),
                    style=self.prompt_color,
                )
                text.append(" ")
            elif mask == 0:
                text.append(
                    tokenizer.convert_ids_to_tokens(token.item()),
                    style=self.system_color,
                )
                text.append(" ")
            else:
                text.append(
                    tokenizer.convert_ids_to_tokens(token.item()),
                    style=self.model_color,
                )
                text.append(" ")
        print(text)
        if show_legend:
            self.show_colour_legend()

    def show_tokens_detail(self, tokenizer, show_legend=False, to_html=False):
        """
        Print the history tokens with each unique token ID in a consistent color.
        """
        if not is_rich_available():
            warnings.warn("install rich to display tokens")
            return
        if to_html:
            html_content = "<p>"
        else:
            text = Text()
        # encoded = tokenizer.encode(text)

        decoded_strings = incremental_decode(tokenizer, self.tokens.tolist())
        for i, (token_text, mask) in enumerate(zip(decoded_strings, self.token_masks)):
            token_id = self.tokens[i]
            color = self.get_color_for_token_id(token_id)
            # text.append(token_text, style=f"bold on {color}")
            if to_html:
                html_content += (
                    f'<span style="background-color:{color}; color:black; font-weight:bold;">{token_text}</span> '
                )
            else:
                text.append(token_text, style=f"bold black on {color}")
                text.append(" ")
        if to_html:
            # console = Console()
            # with console.capture() as capture:
            #     console.print(text)
            # html = capture.get()
            html_content += "</p>"
            # display(HTML(html_content))
            return html_content
        else:
            print(text)
        if not to_html:
            if show_legend:
                self.show_colour_legend()
